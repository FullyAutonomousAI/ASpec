# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common-weight cuDNN sample-budget loss/gradient check; launch with torchrun.

Compare one real small DSpark model's packed CP execution against its independent
document forwards. Inputs are synthetic correctness fixtures, not performance data.
BF16 precision is retained; no optimizer or learning-rate adjustment is made.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.data.utils import DFlashPackingCollator
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.dspark import DSparkModel
from aspec.models.ops.anchors import sample_document_anchor_positions
from aspec.utils.distributed import init_direct_cp_groups


def run(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cp = dist.get_world_size()
    torch.manual_seed(71)
    config = DSparkConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                         vocab_size=257, num_target_layers=1, target_hidden_size=512,
                         target_num_hidden_layers=4, target_layer_ids=[1],
                         mask_token_id=256, markov_rank=32)

    def model(size, scope):
        return DSparkModel(DSparkDraftModel(config), block_size=7, num_anchors=8,
                           anchor_sampling_scope=scope, attention_backend="cudnn_flex_attention",
                           draft_context_parallel_size=size).cuda().bfloat16()

    reference = model(1, "row")  # Construct before CP groups: truly unsharded reference.
    init_direct_cp_groups(cp)
    actual = model(cp, "sample")
    actual.load_state_dict(reference.state_dict())
    # Six documents also force multiple head chunks on CP4 (48 / 4 > 8).
    lengths = (96, 137, 44, 89, 65, 111)
    items = []
    for index, length in enumerate(lengths):
        item = {"input_ids": torch.randint(0, 257, (1, length), device="cuda"),
                "loss_mask": torch.ones(1, length, device="cuda"),
                "hidden_states": torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16),
                "last_hidden_states": torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16),
                "_aspec_work": {"work_id": str(index), "execution_id": str(index)}}
        item["loss_mask"][:, :5] = 0
        items.append(item)
    full = DFlashPackingCollator(sum(lengths), pad_to_max_length=False)(items)
    anchors, keep = sample_document_anchor_positions(
        full["loss_mask"], full["ctx_doc_ids"], lengths, 8, 7,
        torch.Generator(device="cuda").manual_seed(29), alignment=cp,
    )
    # Explicitly exercise the last native candidate: DSpark's final label must
    # be masked, not read from the following packed document.
    offset = 0
    for index, length in enumerate(lengths):
        anchors[0, index * 8 + 7] = offset + length - 7
        offset += length
    shard_items = []
    for item, length in zip(items, lengths, strict=True):
        left, right = length * rank // cp, length * (rank + 1) // cp
        shard_items.append({**item, "hidden_states": item["hidden_states"][:, left:right],
                            "last_hidden_states": item["last_hidden_states"][:, left:right]})
    batch = DFlashPackingCollator(sum(lengths), pad_to_max_length=False,
                                 context_parallel_rank=rank, context_parallel_size=cp)(shard_items)
    head = torch.randn(257, 512, device="cuda", dtype=torch.bfloat16) * 0.03
    out = actual(input_ids=batch["input_ids"], hidden_states_list=batch["hidden_states"],
                 loss_mask=batch["loss_mask"], lm_head_weight=head,
                 last_hidden_states=batch["last_hidden_states"], ctx_doc_ids=batch["ctx_doc_ids"],
                 base_position_ids=batch["base_position_ids"],
                 packed_context_positions=batch.get("packed_context_positions"),
                 packed_document_lengths=lengths, injected_anchors=anchors, injected_keep_mask=keep)
    totals = torch.stack([out[6][0].detach().float(), out[6][1].detach().float()])
    dist.all_reduce(totals)
    (out[6][0] / totals[1]).backward()
    for parameter in actual.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
    reference_terms = []
    offset = 0
    for index, item in enumerate(items):
        ref = reference(input_ids=item["input_ids"], hidden_states_list=item["hidden_states"],
                        loss_mask=item["loss_mask"], lm_head_weight=head,
                        last_hidden_states=item["last_hidden_states"],
                        injected_anchors=anchors[:, index * 8:(index + 1) * 8] - offset,
                        injected_keep_mask=keep[:, index * 8:(index + 1) * 8])
        reference_terms.append(ref[6])
        offset += item["input_ids"].shape[1]
    ref_num = sum(term[0] for term in reference_terms)
    ref_den = sum(term[1] for term in reference_terms)
    ref_loss = ref_num / ref_den
    ref_loss.backward()
    ref_loss = ref_loss.detach()
    gradient = torch.cat([p.grad.float().flatten() for p in actual.parameters() if p.grad is not None])
    ref_gradient = torch.cat([p.grad.float().flatten() for p in reference.parameters() if p.grad is not None])
    loss = totals[0] / totals[1]
    record = {"cp": cp, "precision": "bf16", "backend": "cudnn_flex_attention",
              "fixture": "synthetic small DSpark, not Qwen3-8B performance",
              "loss": float(loss), "reference_loss": float(ref_loss),
              "loss_absolute_diff": float((loss - ref_loss).abs()),
              "loss_relative_diff": float((loss - ref_loss).abs() / ref_loss.abs()),
              "denominator": float(totals[1]), "reference_denominator": float(ref_den),
              "gradient_relative_l2": float((gradient - ref_gradient).norm() / ref_gradient.norm()),
              "gradient_max_absolute_diff": float((gradient - ref_gradient).abs().max())}
    if rank == 0:
        Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    torch.testing.assert_close(totals[1], ref_den)
    assert record["loss_relative_diff"] < 0.003, record
    assert record["gradient_relative_l2"] < 0.03, record
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    run(parser.parse_args())
