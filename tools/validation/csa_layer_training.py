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

"""Profile a real-shape CSA layer forward/backward and its phase gradients."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from aspec.models.draft.csa import (
    CSAForwardMetadata,
    CSALayer,
    build_csa_compression_layout,
    csa_eligible_groups,
)
from aspec.models.draft.dflash import DFlashConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("dense", "indexer_warmup", "sparse"), required=True
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    raw = json.loads(args.config.read_text())
    raw["num_hidden_layers"] = 1
    raw["layer_types"] = ["compressed_sparse_attention"]
    raw["compress_ratios"] = [int(raw["compress_ratios"][0])]
    raw["csa_training_stage"] = args.stage
    config = DFlashConfig(**raw)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    layer = CSALayer(config, 0).to(device=device, dtype=torch.bfloat16)
    layer.keep_projection_fp32()
    if args.stage == "dense":
        layer.indexer.requires_grad_(False)
    elif args.stage == "indexer_warmup":
        layer.requires_grad_(False)
        layer.configure_trainable_parameters()

    context = torch.randn(
        1,
        args.context_length,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.stage != "indexer_warmup",
    )
    draft_length = args.num_anchors * args.block_size
    draft = torch.randn(
        1,
        draft_length,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=args.stage != "indexer_warmup",
    )
    query = torch.randn(
        1,
        draft_length,
        config.num_attention_heads,
        config.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    anchors = torch.arange(args.num_anchors, device=device).view(1, -1)
    anchors = anchors * max(args.context_length // args.num_anchors, 1)
    anchors.clamp_(max=args.context_length - 1)
    metadata = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=args.block_size,
        context_window=int(config.sliding_window),
    )
    context_positions = torch.arange(args.context_length, device=device).view(1, -1)
    offsets = torch.arange(args.block_size, device=device).view(1, 1, -1)
    draft_positions = (anchors.unsqueeze(-1) + offsets).view(1, -1)
    layout = build_csa_compression_layout(
        metadata, context_positions, config.compress_ratios[0]
    )
    eligible = csa_eligible_groups(metadata, layout)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    compressed_key, compressed_value, _, indexer_loss, _ = layer(
        draft_hidden=draft,
        context_hidden=context,
        draft_position_ids=draft_positions,
        context_position_ids=context_positions,
        metadata=metadata,
        main_query=query,
        scale=config.head_dim**-0.5,
        layout=layout,
        eligible=eligible,
    )
    if args.stage == "indexer_warmup":
        loss = indexer_loss.mean()
    else:
        loss = compressed_key.float().square().mean()
        loss = loss + compressed_value.float().square().mean()
        if args.stage == "sparse":
            loss = loss + indexer_loss.mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    def gradient_count(module: torch.nn.Module) -> int:
        return sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "stage": args.stage,
        "shape": {
            "context": args.context_length,
            "draft_queries": draft_length,
            "hidden": config.hidden_size,
            "query_heads": config.num_attention_heads,
            "head_dim": config.head_dim,
            "compressed_groups": layout.num_groups,
        },
        "elapsed_seconds": elapsed,
        "baseline_allocated_mib": baseline / 2**20,
        "peak_allocated_mib": peak / 2**20,
        "peak_delta_mib": (peak - baseline) / 2**20,
        "loss": float(loss.detach()),
        "compressor_finite_gradient_tensors": gradient_count(layer.compressor),
        "indexer_finite_gradient_tensors": gradient_count(layer.indexer),
        "context_gradient": context.grad is not None,
        "draft_gradient": draft.grad is not None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
