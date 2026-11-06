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

"""Native 16-step THD+packing+CP training audit for structured/algorithm drafts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from aspec.models.dflash2 import DFlash2Model
from aspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from aspec.models.draft.dspark import (
    DSparkConfig,
    DSparkDraftModel,
    K3DSparkConfig,
    K3DSparkModel,
)
from aspec.models.dspark import DSparkModel
from aspec.models.ops.direct_cp import build_context_positions
from aspec.training.dflash_trainer import DFlashTrainer
from aspec.training.fsdp import apply_fsdp2, fsdp2_load_full_state_dict
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.distributed import init_direct_cp_groups


def _config(kind: str):
    common = dict(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_target_layers=2,
        target_hidden_size=32,
        target_num_hidden_layers=6,
        target_layer_ids=[1, 3],
        mask_token_id=63,
        max_position_embeddings=128,
        markov_rank=8,
    )
    if kind.startswith("dflash2"):
        common.pop("markov_rank")
        return DFlash2Config(**common, head_dim=128, block_size=2,
                             conv_group_size=4, selector_rank=8, selector_top_k=4,
                             layer_types=["sliding_attention"] * 2 if "swa" in kind else None,
                             sliding_window=4 if "swa" in kind else None,
                             is_causal="noncausal" not in kind if kind != "dflash2" else False)
    if kind == "csa":
        return DSparkConfig(
            **common,
            head_dim=128,
            sliding_window=4,
            layer_types=["compressed_sparse_attention"] * 2,
            compress_ratios=[4, 4],
            csa_training_stage="sparse",
            index_topk=2,
            index_n_heads=2,
            index_head_dim=128,
            csa_workspace_bytes=16 * 1024 * 1024,
        )
    if kind == "qsa":
        return DSparkConfig(
            **common,
            head_dim=128,
            layer_types=["qwen_sparse_attention"] * 2,
            qsa_training_stage="sparse",
            qsa_token_budget=4,
            qsa_compress_ratio=2,
            qsa_index_n_heads=2,
            qsa_index_head_dim=128,
            qsa_rope_dim=64,
            qsa_workspace_bytes=16 * 1024 * 1024,
        )
    if kind == "dsa":
        return DSparkConfig(
            **common, head_dim=128, layer_types=["deepseek_sparse_attention"] * 2,
            dsa_training_stage="sparse", dsa_token_budget=4,
            dsa_index_n_heads=2, dsa_index_head_dim=128, dsa_q_lora_rank=16,
            dsa_rope_dim=64, dsa_workspace_bytes=16 * 1024 * 1024,
        )
    if kind == "gqa":
        return DSparkConfig(**common, head_dim=128)
    return K3DSparkConfig(
        **(common | {"num_key_value_heads": 2}),
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gqa", "csa", "qsa", "dsa", "mla", "dflash2", "dflash2_causal",
                                                 "dflash2_swa", "dflash2_swa_noncausal"), required=True)
    parser.add_argument(
        "--backend", choices=("fa4", "cudnn_flex_attention"), default="fa4"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cudnn_flex_attention" and args.kind in {"csa", "qsa", "dsa"}:
        parser.error("cuDNN THD does not implement selected-entry sparse attention")

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    init_direct_cp_groups(size)
    torch.manual_seed(42)

    config = _config(args.kind)
    draft_class = (DFlash2DraftModel if args.kind.startswith("dflash2") else
                   {"mla": K3DSparkModel}.get(args.kind, DSparkDraftModel))
    draft = draft_class(config).cuda().bfloat16()
    if args.kind == "csa":
        draft.keep_csa_projection_precision()
    draft.freeze_embedding()
    wrapper_class = DFlash2Model if args.kind.startswith("dflash2") else DSparkModel
    wrapper = wrapper_class(
        draft,
        block_size=2,
        num_anchors=8,
        attention_backend=args.backend,
        draft_context_parallel_size=size,
    )
    state = wrapper.state_dict() if rank == 0 else {}
    wrapped = apply_fsdp2(wrapper, args=SimpleNamespace(fsdp_strategy="REPLICATE"))
    wrapped = fsdp2_load_full_state_dict(wrapped, state, None, None)
    optimizer = BF16Optimizer(
        draft,
        lr=6e-4,
        min_lr=6e-5,
        max_grad_norm=1.0,
        total_steps=32,
        warmup_ratio=0.04,
    )
    trainer = SimpleNamespace(optimizer=optimizer, dp_group=dist.group.WORLD)
    device = torch.device("cuda", rank)
    positions = torch.cat(
        (
            build_context_positions(32, rank, size),
            build_context_positions(32, rank, size) + 32,
        )
    ).to(device)
    documents = torch.tensor([[0] * 32 + [1] * 32], device=device)
    rope_positions = (torch.arange(64, device=device) % 32)[None]
    head = torch.randn(64, 32, device=device, dtype=torch.bfloat16)
    rows = []
    for step in range(1, 17):
        torch.manual_seed(100 + step)
        features = [
            torch.randn(1, 64, 32, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        last = torch.randn(1, 64, 32, device=device, dtype=torch.bfloat16)
        values = dict(
            input_ids=torch.randint(0, 63, (1, 64), device=device),
            loss_mask=torch.ones(1, 64, device=device),
            hidden_states_list=[value.index_select(1, positions) for value in features],
            last_hidden_states=last.index_select(1, positions),
            lm_head_weight=head,
            ctx_doc_ids=documents,
            base_position_ids=rope_positions,
            injected_anchors=torch.tensor(
                [[1, 9, 17, 25, 33, 41, 49, 57]], device=device
            ),
            injected_keep_mask=torch.ones(
                1, 8, device=device, dtype=torch.bool
            ),
            packed_context_positions=positions[None],
        )
        optimizer.zero_grad()
        output = wrapped(**values)
        numerator, denominator = output[6]
        DFlashTrainer._backward(trainer, numerator)
        observations = wrapper.attention_policy_observed["layers"]
        if len(observations) != config.num_hidden_layers:
            raise RuntimeError(
                "Draft attention evidence is incomplete: "
                f"observed {len(observations)} of {config.num_hidden_layers} layers; "
                f"observations={observations!r}"
            )
        assert all(
            observation["qkv_format"] == "thd"
            and observation["context_parallel_size"] == size
            and observation.get("kernel_backend", "fa4") == args.backend
            and observation["backward_observed"]
            for observation in observations
        )
        total = DFlashTrainer._normalize_accumulated_gradients(
            trainer, denominator, 1
        )
        loss = numerator.detach().float().clone()
        dist.all_reduce(loss)
        norm = optimizer.step()
        assert torch.isfinite(loss) and torch.isfinite(norm)
        for parameter in draft.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(expected, parameter)
        rows.append(
            {
                "step": step,
                "loss": (loss / total).item(),
                "gradient_norm": norm.item(),
            }
        )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "synthetic_data": True,
                    "kind": args.kind,
                    "backend": args.backend,
                    "qkv_format": "thd",
                    "packing": True,
                    "cp_size": size,
                    "replicas_equal": True,
                    "steps": rows,
                },
                indent=2,
            )
        )
        print(
            f"PASS {args.kind.upper()} {args.backend} THD packing CP{size}: "
            "16 native updates",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
