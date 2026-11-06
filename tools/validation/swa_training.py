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

"""Small native DSpark/REPLICATE SWA lifecycle audit; synthetic data, no speed claim."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.dspark import DSparkModel
from aspec.models.ops.direct_cp import build_context_positions
from aspec.training.dflash_trainer import DFlashTrainer
from aspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.distributed import init_direct_cp_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fa4", "cudnn_flex_attention", "flex_attention"], default="fa4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    init_direct_cp_groups(size)
    torch.manual_seed(42)
    config = DSparkConfig(
        hidden_size=32, intermediate_size=64, vocab_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, num_target_layers=2,
        target_hidden_size=32, target_num_hidden_layers=6, target_layer_ids=[1, 3],
        mask_token_id=63, max_position_embeddings=128, markov_rank=8, sliding_window=3,
        layer_types=["full_attention", "sliding_attention"],
    )
    draft = DSparkDraftModel(config).cuda().bfloat16()
    draft.freeze_embedding()
    wrapper = DSparkModel(draft, block_size=2, num_anchors=8, attention_backend=args.backend,
                          draft_context_parallel_size=size)
    state = wrapper.state_dict() if rank == 0 else {}
    wrapped = apply_fsdp2(wrapper, args=SimpleNamespace(fsdp_strategy="REPLICATE"))
    wrapped = fsdp2_load_full_state_dict(wrapped, state, None, None)
    optimizer = BF16Optimizer(draft, lr=6e-4, min_lr=6e-5, max_grad_norm=1.0,
                               total_steps=32, warmup_ratio=.04)
    trainer = SimpleNamespace(optimizer=optimizer, dp_group=dist.group.WORLD)
    device = torch.device("cuda", rank)
    positions = torch.cat([build_context_positions(32, rank, size),
                           build_context_positions(32, rank, size) + 32]).to(device)
    docs = torch.tensor([[0]*32 + [1]*32], device=device)
    head = torch.randn(64, 32, device=device, dtype=torch.bfloat16)
    rows = []
    for step in range(1, 17):
        torch.manual_seed(100 + step)
        features = [torch.randn(1, 64, 32, device=device, dtype=torch.bfloat16) for _ in range(2)]
        last = torch.randn(1, 64, 32, device=device, dtype=torch.bfloat16)
        values = dict(
            input_ids=torch.randint(0, 63, (1, 64), device=device),
            loss_mask=torch.ones(1, 64, device=device),
            hidden_states_list=[v.index_select(1, positions) for v in features],
            last_hidden_states=last.index_select(1, positions), lm_head_weight=head,
            ctx_doc_ids=docs, base_position_ids=(torch.arange(64, device=device) % 32)[None],
            injected_anchors=torch.tensor([[1, 9, 17, 25, 33, 41, 49, 57]], device=device),
            injected_keep_mask=torch.ones(1, 8, device=device, dtype=torch.bool),
        )
        if size > 1:
            values["packed_context_positions"] = positions[None]
        optimizer.zero_grad()
        output = wrapped(**values)
        assert wrapper.attention_windows_observed == [None, 3], (
            wrapper.attention_windows_observed, config.layer_types,
            draft.config.layer_types, draft.config.sliding_window,
        )
        numerator, denominator = output[6]
        DFlashTrainer._backward(trainer, numerator)
        total = DFlashTrainer._normalize_accumulated_gradients(trainer, denominator, 1)
        loss = numerator.detach().float().clone()
        dist.all_reduce(loss)
        norm = optimizer.step()
        assert torch.isfinite(loss) and torch.isfinite(norm)
        for p in draft.parameters():
            expected = p.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(expected, p)
        rows.append({"step": step, "loss": (loss/total).item(), "gradient_norm": norm.item(),
                     "observed_context_windows": list(wrapper.attention_windows_observed)})
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"synthetic_data": True, "backend": args.backend,
                                          "cp_size": size, "replicas_equal": True,
                                          "steps": rows}, indent=2))
        print(f"PASS mixed full/SWA {args.backend} CP{size}: 16 native DSpark updates", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
