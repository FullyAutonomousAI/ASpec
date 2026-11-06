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

"""16-state native MTP: document SDPA reference, packed FA4 and CP self comparison."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.data.utils import DFlashPackingCollator
from aspec.inference.context import context_shard_bounds
from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel
from aspec.models.mtp import MTPModel, mtp_step_weights
from aspec.models.ops.eagle_thd import EagleTHDLayout
from aspec.training.optimizer import BF16Optimizer
from tools.validation.eagle_thd_reference import difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--reference-backend", choices=("sdpa", "fa4"), default="sdpa")
    parser.add_argument("--common-states", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.steps < 16:
        raise ValueError("use a fresh output and at least16 steps")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    native = {
        "model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
        "vocab_size": 97, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "rms_norm_eps": 1e-6, "attention_dropout": 0.0, "attention_bias": False,
        "hidden_act": "silu", "max_position_embeddings": 128, "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                            "partial_rotary_factor": 0.25, "mrope_section": [4, 6, 6],
                            "mrope_interleaved": True},
    }
    torch.manual_seed(47)
    config = NativeMTPConfig(native_config=native)
    draft = NativeMTPDraftModel(config, attention_backend="fa4").cuda().bfloat16()
    ref_draft = NativeMTPDraftModel(config, attention_backend=args.reference_backend).cuda().bfloat16()
    ref_draft.load_state_dict(draft.state_dict())
    model, reference = MTPModel(draft, length=3), MTPModel(ref_draft, length=3)
    settings = dict(lr=1e-4, total_steps=32, warmup_ratio=0.0, weight_decay=0.0)
    optimizer, ref_optimizer = BF16Optimizer(draft, **settings), BF16Optimizer(ref_draft, **settings)
    weights = mtp_step_weights(3, 0.6)
    actual_params = {n: p for n, p in draft.named_parameters() if p.requires_grad}
    ref_params = dict(ref_draft.named_parameters())
    records = []
    try:
        for step in range(args.steps):
            if args.common_states:
                ref_draft.load_state_dict(draft.state_dict())
                ref_optimizer.sync_fp32_params_from_model()
                for destination, source in zip(ref_optimizer.fp32_params, optimizer.fp32_params, strict=True):
                    destination.data.copy_(source.data)
                ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                ref_optimizer.scheduler.load_state_dict(optimizer.scheduler.state_dict())
            torch.manual_seed(79 + step)
            lengths = [17, 11, 2]
            ids = torch.randint(0, 97, (3, 17), device="cuda")
            hidden = torch.randn(3, 17, 32, dtype=torch.bfloat16, device="cuda")
            mask = torch.zeros_like(ids)
            attention = torch.zeros_like(ids)
            samples = []
            for doc, length in enumerate(lengths):
                mask[doc, 3:length] = 1
                attention[doc, :length] = 1
                lo, hi = context_shard_bounds(length, rank, size)
                samples.append({"input_ids": ids[doc:doc + 1, :length],
                                "loss_mask": mask[doc:doc + 1, :length],
                                "hidden_states": hidden[doc:doc + 1, lo:hi],
                                "_aspec_work": {"execution_id": f"doc-{doc}"}})
            packed = DFlashPackingCollator(64, pad_to_max_length=False,
                        context_parallel_rank=rank, context_parallel_size=size)(samples)
            layout, local, _ = EagleTHDLayout.from_batch(packed, cp_rank=rank,
                        cp_size=size, group=dist.group.WORLD, teacher_required=False)
            actual = model(None, local, None, None, thd_layout=layout)
            sum(w * v for w, v in zip(weights, actual[0], strict=True)).backward()
            if size > 1:
                for parameter in actual_params.values():
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(size)
            if args.reference_backend == "sdpa":
                expected = reference(ids, hidden, mask, attention)
            else:
                full, local, _ = EagleTHDLayout.from_batch({"input_ids": ids, "hidden_states": hidden,
                        "loss_mask": mask, "attention_mask": attention}, teacher_required=False)
                expected = reference(None, local, None, None, thd_layout=full)
            sum(w * v for w, v in zip(weights, expected[0], strict=True)).backward()
            row = {"step": step + 1, "loss": difference(actual[1], expected[1]),
                   "gradients": difference([p.grad for p in actual_params.values()],
                                           [ref_params[n].grad for n in actual_params]),
                   "per_tensor_gradients": {n: difference([p.grad], [ref_params[n].grad])
                                            for n, p in actual_params.items()}}
            optimizer.step()
            ref_optimizer.step()
            row["model"] = difference(list(actual_params.values()), [ref_params[n] for n in actual_params])
            row["master"] = difference(optimizer.fp32_params, ref_optimizer.fp32_params)
            row["moments"] = {key: difference([optimizer.state[p][key] for p in optimizer.fp32_params],
                        [ref_optimizer.state[p][key] for p in ref_optimizer.fp32_params])
                              for key in ("exp_avg", "exp_avg_sq")}
            records.append(row)
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            if rank == 0:
                print(step + 1, row["loss"], row["gradients"], flush=True)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"cp": size, "synthetic_features": True,
                    "reference": args.reference_backend, "common_states": args.common_states,
                    "records": records}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
