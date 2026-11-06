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

"""Real GPU MoE/REPLICATE accumulation and CP-router audit, not a benchmark.

Run with torchrun --standalone --nproc-per-node=2 -m
tools.validation.moe_draft_distributed --output /path/to/report.json.
This isolates FFN/distributed mathematics; the recipe smoke covers the real
Target -> data plane -> complete DSpark model and checkpoint lifecycle.
"""

import argparse
import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.ling_moe import LingDraftMoE, ling_router_terms, update_ling_bias
from aspec.models.draft.moe import QwenDraftMoE, router_loss_terms
from aspec.training.dflash_trainer import DFlashTrainer
from aspec.training.fsdp import apply_fsdp2
from aspec.training.optimizer import BF16Optimizer


def error(actual, expected):
    delta = (actual.float() - expected.float()).flatten()
    denominator = expected.float().norm().clamp_min(1e-12)
    return {"abs_max": delta.abs().max().item(), "relative_l2": (delta.norm() / denominator).item()}


def sample(rank, micro, step, dtype):
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank * 97 + micro * 11 + step)
    # Different ranks/micro-batches have different valid counts and routing.
    x = torch.randn(1, 6, 16, device="cuda", dtype=dtype, generator=generator)
    target = torch.randn(x.shape, device="cuda", dtype=dtype, generator=generator)
    mask = torch.arange(6, device="cuda") < (2 + (rank + micro + step) % 5)
    return x, target, mask.float()


def terms(output, target, mask, ffn_type="qwen3_moe", counts_out=None):
    hidden, logits = output
    denominator = mask.sum()
    primary = ((hidden.float() - target.float()).square().mean(-1) * mask).sum()
    if ffn_type == "ling_moe":
        auxiliary, components = ling_router_terms((logits,), mask, denominator, 0.0000035)
        if counts_out is not None:
            counts_out.append(components["_ling_router_counts"])
        return primary + auxiliary, denominator
    auxiliary, _ = router_loss_terms((logits,), mask, denominator, 2)
    return primary + 0.01 * auxiliary, denominator


def ling_config():
    return DFlashConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        ffn_type="ling_moe", num_experts=8, num_experts_per_tok=2,
                        moe_intermediate_size=12, num_shared_experts=1,
                        moe_shared_expert_intermediate_size=10, n_group=2, topk_group=1)


def run_replicate(dtype, steps, ffn_type="qwen3_moe"):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(37)
    config = DFlashConfig(hidden_size=16, intermediate_size=32, ffn_type="qwen3_moe",
                          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12)
    if ffn_type == "ling_moe":
        config = ling_config()
    model = (LingDraftMoE(config) if ffn_type == "ling_moe" else QwenDraftMoE(config)).cuda().to(dtype)
    reference = copy.deepcopy(model)
    # Exercise the exact wrapper used by the production trainer, not a second DDP setup.
    wrapped = apply_fsdp2(model, args=SimpleNamespace(fsdp_strategy="REPLICATE"))
    optimizer = BF16Optimizer(model, lr=0.001, max_grad_norm=1000, total_steps=steps + 2, warmup_ratio=0)
    ref_optimizer = BF16Optimizer(reference, lr=0.001, max_grad_norm=1000, total_steps=steps + 2, warmup_ratio=0)
    trainer = object.__new__(DFlashTrainer)
    trainer.optimizer, trainer.dp_group = optimizer, dist.group.WORLD
    rows = []
    for step in range(steps):
        local_den = torch.zeros((), device="cuda")
        local_counts, reference_counts = [], []
        # Two actual accumulation forwards. replicate's normal reduction is
        # intentionally exercised even when an expert is unused on a rank.
        for micro in range(2):
            x, target, mask = sample(rank, micro, step, dtype)
            numerator, denominator = terms(wrapped(x), target, mask, ffn_type, local_counts)
            (numerator / 2).backward()
            local_den += denominator
        global_den = trainer._normalize_accumulated_gradients(local_den, 2)
        # Serial reference of the SAME rank-local micro-batch objectives.
        for source_rank in range(world):
            for micro in range(2):
                x, target, mask = sample(source_rank, micro, step, dtype)
                numerator, _ = terms(reference(x), target, mask, ffn_type, reference_counts)
                (numerator / global_den).backward()
        p = list(model.parameters())
        q = list(reference.parameters())
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p)
        gradients = error(torch.cat([v.grad.flatten() for v in p]),
                          torch.cat([v.grad.flatten() for v in q]))
        tolerance = 2e-5 if dtype == torch.float32 else 0.025
        assert gradients["relative_l2"] < tolerance, gradients
        optimizer.step()
        ref_optimizer.step()
        if ffn_type == "ling_moe":
            update_ling_bias([model], torch.stack(local_counts).sum(0), .001, group=dist.group.WORLD)
            update_ling_bias([reference], torch.stack(reference_counts).sum(0), .001)
            torch.testing.assert_close(model.gate.expert_bias, reference.gate.expert_bias, atol=0, rtol=0)
            expected_bias = model.gate.expert_bias.clone()
            dist.broadcast(expected_bias, 0)
            assert torch.equal(model.gate.expert_bias, expected_bias)
        weights = error(torch.cat([v.flatten() for v in p]), torch.cat([v.flatten() for v in q]))
        assert weights["relative_l2"] < (2e-5 if dtype == torch.float32 else 0.025), weights
        # All replicas must remain exactly identical after every optimizer update.
        for parameter in p:
            expected = parameter.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(parameter, expected)
        rows.append({"step": step + 1, "gradients": gradients, "weights": weights})
    return {"dtype": str(dtype), "steps": rows, "replicas_equal_each_step": True}


def run_cp_router(ffn_type="qwen3_moe"):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(73)
    complete = torch.randn(world * 5, 4, device="cuda")
    mask = torch.ones(world * 5, device="cuda")
    # One CP owner has no valid tokens; others must still train correctly.
    mask[:5] = 0
    local = complete[rank * 5:(rank + 1) * 5].clone().requires_grad_(True)
    local_mask = mask[rank * 5:(rank + 1) * 5]
    if ffn_type == "ling_moe":
        selected = complete.sigmoid().topk(2, dim=-1).indices
        numerator, _ = ling_router_terms(((local, selected[rank * 5:(rank + 1) * 5]),),
                                         local_mask, local_mask.sum(), .01, cp_group=dist.group.WORLD)
    else:
        numerator, _ = router_loss_terms((local,), local_mask, local_mask.sum(), 2,
                                        cp_group=dist.group.WORLD)
    (numerator / mask.sum()).backward()
    reference = complete.clone().requires_grad_(True)
    if ffn_type == "ling_moe":
        total, _ = ling_router_terms(((reference, selected),), mask, mask.sum(), .01)
    else:
        total, _ = router_loss_terms((reference,), mask, mask.sum(), 2)
    (total / mask.sum()).backward()
    expected = reference.grad[rank * 5:(rank + 1) * 5]
    torch.testing.assert_close(local.grad, expected, rtol=2e-5, atol=2e-6)
    return {"relative_l2": error(local.grad, expected)["relative_l2"], "empty_owner": rank == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--ffn-type", choices=["qwen3_moe", "ling_moe"], default="qwen3_moe")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    import os

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        result = {"claim": "MoE FFN distributed correctness, not full attention/CP acceptance",
                  "world_size": dist.get_world_size(), "ffn_type": args.ffn_type, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(),
                  "replicate": [run_replicate(torch.float32, args.steps, args.ffn_type),
                                run_replicate(torch.bfloat16, args.steps, args.ffn_type)],
                  "cp_router": run_cp_router(args.ffn_type)}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, result)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(gathered, indent=2) + "\n")
            print(f"MoE distributed correctness passed: {args.output}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
