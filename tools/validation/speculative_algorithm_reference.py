#!/usr/bin/env python3
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

"""Bounded EAGLE3 BF16 loss/gradient/update comparison with fixed TorchSpec."""

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from aspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from aspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded
from aspec.training.optimizer import BF16Optimizer


def difference(actual, expected):
    a = torch.cat([v.detach().float().reshape(-1) for v in actual])
    b = torch.cat([v.detach().float().reshape(-1) for v in expected])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite reference comparison")
    return {
        "max_absolute_diff": (a - b).abs().max().item(),
        "relative_l2_diff": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_root))
    reference = importlib.import_module("torchspec.models.eagle3")
    reference_draft = importlib.import_module("torchspec.models.draft.llama3_eagle")
    reference_optimizer = importlib.import_module("torchspec.training.optimizer")
    torch.cuda.set_device(0)
    records = []
    with tempfile.TemporaryDirectory(prefix="aspec-eagle3-reference-") as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        try:
            for loss_type in ("forward_kl", "lk_alpha", "lk_lambda"):
                torch.manual_seed(29)
                config = LlamaConfig(
                    hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    vocab_size=97, draft_vocab_size=97, target_hidden_size=64,
                    max_position_embeddings=256, pad_token_id=0,
                    rms_norm_eps=1e-6, pretraining_tp=1, num_aux_hidden_states=3,
                )
                draft = LlamaForCausalLMEagle3(config, attention_backend="sdpa").cuda().bfloat16()
                draft.freeze_embedding()
                ref_draft = reference_draft.LlamaForCausalLMEagle3(
                    config, attention_backend="sdpa"
                ).cuda().bfloat16()
                ref_draft.freeze_embedding()
                ref_draft.load_state_dict(draft.state_dict())
                actual = Eagle3Model(draft, length=3, attention_backend="sdpa", loss_type=loss_type)
                expected = reference.Eagle3Model(
                    ref_draft, length=3, attention_backend="sdpa", loss_type=loss_type,
                )
                common = dict(lr=6e-4, weight_decay=0.0, total_steps=32, warmup_ratio=0.0)
                optimizer = BF16Optimizer(draft, **common)
                ref_optimizer = reference_optimizer.BF16Optimizer(ref_draft, **common)
                weights = (1.0, 0.8, 0.64)
                for step in range(args.steps):
                    torch.manual_seed(103 + step)
                    hidden = torch.randn(2, 29, 192, device="cuda", dtype=torch.bfloat16)
                    teacher = torch.randn(2, 29, 64, device="cuda", dtype=torch.bfloat16)
                    head = torch.randn(97, 64, device="cuda", dtype=torch.bfloat16) * 0.1
                    mask = torch.ones(2, 29, device="cuda")
                    mask[0, :7] = 0
                    mask[1, :13] = 0
                    inputs = dict(
                        input_ids=torch.randint(0, 97, (2, 29), device="cuda"),
                        attention_mask=torch.ones(2, 29, device="cuda", dtype=torch.long),
                        hidden_states=hidden, loss_mask=mask,
                    )
                    out = actual(**inputs, target=compute_lazy_target_padded(teacher, head, 3))
                    ref_out = expected(
                        **inputs, target=reference.compute_lazy_target_padded(teacher, head, 3),
                    )
                    loss = sum(w * value for w, value in zip(weights, out[0], strict=True))
                    ref_loss = sum(w * value for w, value in zip(weights, ref_out[0], strict=True))
                    loss.backward()
                    ref_loss.backward()
                    gradients = difference(
                        [p.grad for p in draft.parameters() if p.requires_grad],
                        [p.grad for p in ref_draft.parameters() if p.requires_grad],
                    )
                    before = [p.detach().clone() for p in optimizer.fp32_params]
                    ref_before = [p.detach().clone() for p in ref_optimizer.fp32_params]
                    norm = optimizer.step()
                    ref_norm = ref_optimizer.step()
                    updates = difference(
                        [p - start for p, start in zip(optimizer.fp32_params, before, strict=True)],
                        [p - start for p, start in zip(ref_optimizer.fp32_params, ref_before, strict=True)],
                    )
                    record = {
                        "loss_type": loss_type, "step": step + 1,
                        "loss": loss.item(), "reference_loss": ref_loss.item(),
                        "loss_absolute_diff": abs(loss.item() - ref_loss.item()),
                        "loss_relative_diff": abs(loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1e-30),
                        "gradient": gradients, "optimizer_update": updates,
                        "gradient_norm_absolute_diff": abs(float(norm) - float(ref_norm)),
                    }
                    if max(record["loss_relative_diff"], gradients["relative_l2_diff"],
                           updates["relative_l2_diff"]) > 1e-4:
                        raise AssertionError(record)
                    records.append(record)
                    optimizer.zero_grad()
                    ref_optimizer.zero_grad()
        finally:
            dist.destroy_process_group()
    payload = {
        "claim": "BF16 EAGLE3 fixed-reference numerical correctness, not model quality or throughput",
        "torchspec_revision": "6c042a87140a84d13839e341ece2c5c3ada918bc",
        "dtype": "bfloat16", "backend": "sdpa", "steps_per_objective": args.steps,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"records": len(records), "max_loss_relative_diff":
                      max(r["loss_relative_diff"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
