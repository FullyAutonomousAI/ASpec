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

"""Real replicated candidate updates with one empty DP owner; not a benchmark."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from aspec.config.eagle4 import Eagle4Config
from aspec.models.draft.eagle4 import Eagle4DraftModel
from aspec.models.eagle3 import LazyTarget
from aspec.models.eagle4 import Eagle4TrainingModel
from aspec.training.eagle4_trainer import Eagle4Trainer
from aspec.training.fsdp import apply_fsdp2
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.tensor import padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    if size < 2:
        raise ValueError("Use at least two training ranks")
    torch.manual_seed(42)
    cfg = Eagle4Config(
        architectures=["LlamaForCausalLMEagle4"], hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=128, vocab_size=97,
        num_aux_hidden_states=2, target_hidden_size=32, pad_token_id=0,
        max_position_embeddings=128, use_rope=False, fc_norm=True, norm_output=True,
        layer_types=["sliding_attention"], sliding_window=4,
        efficient_lm_head_config={"type": "low_rank", "rank": 8, "num_candidates": 16},
        candidate_teacher_top_k=8, candidate_loss_chunk_size=3,
        eagle4_training_stage="candidate_head",
    )
    draft = Eagle4DraftModel(cfg, attention_backend="sdpa").cuda().bfloat16()
    draft.configure_training_stage()
    body = {n: p.detach().clone() for n, p in draft.named_parameters() if not p.requires_grad}
    initial_head = {n: p.detach().clone() for n, p in draft.named_parameters() if p.requires_grad}
    wrapper = Eagle4TrainingModel(draft, length=3, attention_backend="sdpa")
    wrapper = apply_fsdp2(wrapper, mesh=init_device_mesh("cuda", (size,)),
                          args=argparse.Namespace(fsdp_strategy="REPLICATE", attention_backend="sdpa"))
    optimizer = BF16Optimizer(draft, lr=6e-4, total_steps=32, warmup_ratio=0)
    trainer = object.__new__(Eagle4Trainer)
    trainer.draft_model, trainer.dp_group = draft, dist.group.WORLD
    records = []
    for step in range(args.steps):
        metrics = []
        for micro in range(2):
            wrapper.set_requires_gradient_sync(micro == 1)
            torch.manual_seed(1000 + step * 8 + rank * 2 + micro)
            ids = torch.randint(1, 97, (1, 11), device="cuda")
            mask = torch.ones_like(ids) if rank else torch.zeros_like(ids)
            result = wrapper(input_ids=padding(ids, left=False), attention_mask=torch.ones_like(ids),
                             target=LazyTarget(None, None), loss_mask=mask,
                             hidden_states=torch.randn(1, 11, 64, device="cuda", dtype=torch.bfloat16))
            sum(w * loss for w, loss in zip((1.0, 0.8, 0.64), result[0], strict=True)).div(2).backward()
            metrics.append({"acc_counts": torch.stack(result[3])})
        norm = optimizer.step()
        trainer._after_optimizer_step(metrics, norm)
        counters = [torch.empty_like(draft.candidate_head_steps) for _ in range(size)]
        dist.all_gather(counters, draft.candidate_head_steps)
        if not all(int(value) == step + 1 for value in counters):
            raise AssertionError(f"Candidate update counters disagree: {counters}")
        if not all(torch.equal(body[n], p) and p.grad is None
                   for n, p in draft.named_parameters() if not p.requires_grad):
            raise AssertionError("Candidate updates changed the frozen Draft body")
        records.append({"step": step + 1, "all_rank_counters": [int(x) for x in counters],
                        "local_supervised_positions": int(sum(m["acc_counts"].sum() for m in metrics))})
    for name, parameter in draft.named_parameters():
        if parameter.requires_grad:
            copies = [torch.empty_like(parameter) for _ in range(size)]
            dist.all_gather(copies, parameter)
            if not all(torch.equal(copies[0], value) for value in copies):
                raise AssertionError(f"Candidate head differs across ranks: {name}")
            if torch.equal(initial_head[name], parameter):
                raise AssertionError(f"Candidate parameter was never updated: {name}")
    if rank == 0:
        args.output.write_text(json.dumps({"synthetic_data": True, "replicas": size,
                                          "frozen_body_exact": True, "heads_equal_and_updated": True,
                                          "records": records}, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
