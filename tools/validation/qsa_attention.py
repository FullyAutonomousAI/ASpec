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

"""Check one real-shape QSA/DSA Draft attention layer on CUDA.

This is a component validation, not an end-to-end throughput claim.  It measures
the per-layer hot path after the compression layout or dense BlockMask shared by
all Draft layers has been prepared.  Indexer/Top-K, sparse core attention, KL
and backward remain inside the measured region.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.draft.dflash import DFlashAttention, DFlashConfig
from aspec.models.draft.qsa import (
    QSAForwardMetadata,
    build_qsa_compression_layout,
)
from aspec.models.ops.flex_attention import compile_friendly_create_block_mask


def _qsa_config(path: Path, stage: str) -> DFlashConfig:
    raw = json.loads(path.read_text())
    raw.update(
        num_hidden_layers=1,
        layer_types=["qwen_sparse_attention"],
        compress_ratios=[0],
        qsa_training_stage=stage,
        qsa_token_budget=2048,
        qsa_compress_ratio=4,
        qsa_index_n_heads=4,
        qsa_index_head_dim=128,
        qsa_rope_dim=64,
        qsa_rope_theta=raw.get("rope_theta", 10000.0),
        qsa_indexer_loss_coef=1.0,
        qsa_workspace_bytes=128 * 1024 * 1024,
    )
    for name in (
        "sliding_window",
        "compress_rope_theta",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "csa_indexer_loss_coef",
        "csa_training_stage",
    ):
        raw.pop(name, None)
    return DFlashConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("qsa", "dsa"), default="qsa")
    parser.add_argument(
        "--stage", choices=("dense", "indexer_warmup", "sparse"), required=True
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.context_length, args.num_anchors, args.block_size) <= 0:
        raise ValueError("context length, anchors and block size must be positive")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations positive")

    if args.kind == "dsa":
        raw = json.loads(args.config.read_text())
        raw.update(num_hidden_layers=1, layer_types=["deepseek_sparse_attention"], compress_ratios=[0],
                   dsa_training_stage=args.stage)
        config = DFlashConfig(**raw)
    else:
        config = _qsa_config(args.config, args.stage)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20261015)

    layer = DFlashAttention(config, 0).to(device=device, dtype=torch.bfloat16)
    layer.set_attention_backend("flex_attention")
    selector = getattr(layer, f"{args.kind}_layer")
    if args.stage == "dense":
        selector.indexer.requires_grad_(False)
    elif args.stage == "indexer_warmup":
        layer.requires_grad_(False)
        selector.indexer.requires_grad_(True)

    draft_length = args.num_anchors * args.block_size
    context = torch.randn(
        1,
        args.context_length,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.stage != "indexer_warmup",
    )
    draft = torch.randn(
        1,
        draft_length,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.stage != "indexer_warmup",
    )
    context_positions = torch.arange(args.context_length, device=device).view(1, -1)
    anchors = torch.linspace(
        1,
        args.context_length - (args.block_size if args.kind == "dsa" else 1),
        args.num_anchors,
        device=device,
    ).long().view(1, -1)
    offsets = torch.arange(args.block_size, device=device).view(1, 1, -1)
    draft_positions = (anchors.unsqueeze(-1) + offsets).view(1, -1)
    metadata = QSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=args.block_size,
    )
    if args.kind == "dsa" and args.stage != "dense":
        from aspec.models.draft.dsa import build_dsa_token_layout

        layout = build_dsa_token_layout(metadata, context_positions, draft_positions,
                                       rope_dim=config.dsa_rope_dim, rope_theta=config.dsa_rope_theta)
    else:
        layout = (build_qsa_compression_layout(metadata, context_positions, ratio=4)
                  if args.stage != "dense" else None)
    dense_mask = None
    if args.stage != "sparse":
        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchors,
            block_keep_mask=metadata.block_keep_mask,
            ctx_len=args.context_length,
            block_size=args.block_size,
        )
        dense_mask = compile_friendly_create_block_mask(
            mask_mod=mask_mod,
            B=1,
            H=None,
            Q_LEN=draft_length,
            KV_LEN=args.context_length + draft_length,
            device=device,
        )

    def iteration() -> tuple[torch.Tensor, int]:
        layer.zero_grad(set_to_none=True)
        if context.grad is not None:
            context.grad = None
        if draft.grad is not None:
            draft.grad = None
        output = layer(
            draft_hidden=draft,
            context_hidden=context,
            draft_position_ids=draft_positions,
            context_position_ids=context_positions,
            block_mask=dense_mask,
            **{f"{args.kind}_metadata": metadata, f"{args.kind}_layout": layout},
        )
        indexer_loss = getattr(layer, f"{args.kind}_indexer_loss_per_query").float().mean()
        loss = indexer_loss
        if args.stage != "indexer_warmup":
            loss = loss + output.float().square().mean()
        loss.backward()
        finite_gradients = sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        return loss.detach(), finite_gradients

    for _ in range(args.warmup):
        iteration()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    elapsed_ms = []
    loss = None
    finite_gradients = 0
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, finite_gradients = iteration()
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    peak = torch.cuda.max_memory_allocated(device)
    indexer_gradients = sum(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in selector.indexer.parameters()
        if parameter.requires_grad
    )
    non_indexer_gradients = sum(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in layer.named_parameters()
        if f"{args.kind}_layer.indexer." not in name and parameter.requires_grad
    )
    result = {
        "component_validation_only": True,
        "kind": args.kind,
        "stage": args.stage,
        "shape": {
            "context": args.context_length,
            "draft_queries": draft_length,
            "hidden": config.hidden_size,
            "query_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "compressed_groups": layout.num_groups if layout is not None and args.kind == "qsa" else None,
            **({"indexed_tokens": layout.num_groups} if layout is not None and args.kind == "dsa" else {}),
            "token_budget": getattr(config, f"{args.kind}_token_budget"),
        },
        "iterations": args.iterations,
        "elapsed_ms": elapsed_ms,
        "median_elapsed_ms": statistics.median(elapsed_ms),
        "baseline_allocated_mib": baseline / 2**20,
        "peak_allocated_mib": peak / 2**20,
        "peak_delta_mib": (peak - baseline) / 2**20,
        "shared_metadata_setup_in_timed_region": False,
        "loss": float(loss),
        "finite_gradient_tensors": finite_gradients,
        "indexer_finite_gradient_tensors": indexer_gradients,
        "non_indexer_finite_gradient_tensors": non_indexer_gradients,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            iteration()
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
