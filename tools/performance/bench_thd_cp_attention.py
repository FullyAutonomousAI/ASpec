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

"""Measure one packed DSpark THD attention forward/backward at a legal CP size."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    build_context_positions,
    direct_cp_attention,
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _anchors(
    *, context_length: int, documents: int, num_anchors: int, block_size: int, device
) -> torch.Tensor:
    if context_length % documents or num_anchors % documents:
        raise ValueError("context_length and num_anchors must divide evenly by documents")
    document_length = context_length // documents
    if document_length <= block_size:
        raise ValueError("each packed document must be longer than block_size")
    per_document = num_anchors // documents
    values = []
    for start in range(0, context_length, document_length):
        values.append(
            torch.linspace(
                start,
                start + document_length - block_size,
                per_document,
                device=device,
                dtype=torch.float64,
            ).round().long()
        )
    return torch.cat(values).view(1, -1)


def _local_context_rows(
    *, context_length: int, documents: int, cp_rank: int, cp_size: int, device
) -> torch.Tensor:
    document_length = context_length // documents
    return torch.cat(
        [
            torch.arange(
                start + document_length * cp_rank // cp_size,
                start + document_length * (cp_rank + 1) // cp_size,
                device=device,
            )
            for start in range(0, context_length, document_length)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fa4", "cudnn_flex_attention"), required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--documents", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1:
        parser.error("warmup and runs must be positive")

    rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cp_size = dist.get_world_size()
    try:
        if args.num_anchors % cp_size:
            raise ValueError("num_anchors must be divisible by CP size")
        device = torch.device("cuda", rank)
        torch.manual_seed(1701)
        anchors = _anchors(
            context_length=args.context_length,
            documents=args.documents,
            num_anchors=args.num_anchors,
            block_size=args.block_size,
            device=device,
        )
        local_blocks = build_context_positions(
            args.num_anchors, rank, cp_size, layout="contiguous"
        ).to(device)
        query_rows = (
            local_blocks[:, None] * args.block_size
            + torch.arange(args.block_size, device=device)[None]
        ).flatten()
        context_rows = _local_context_rows(
            context_length=args.context_length,
            documents=args.documents,
            cp_rank=rank,
            cp_size=cp_size,
            device=device,
        )
        document_length = args.context_length // args.documents
        global_docs = (
            torch.arange(args.documents, device=device)
            .repeat_interleave(document_length)
            .view(1, -1)
        )
        query = torch.randn(
            1, 32, query_rows.numel(), 128,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        context_key = torch.randn(
            1, 8, context_rows.numel(), 128,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        context_value = torch.randn_like(context_key, requires_grad=True)
        draft_key = torch.randn(
            1, 8, query_rows.numel(), 128,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        draft_value = torch.randn_like(draft_key, requires_grad=True)
        inputs = (query, context_key, context_value, draft_key, draft_value)
        dout = torch.randn(
            1, 32, query_rows.numel(), 128,
            device=device, dtype=torch.bfloat16,
        )
        metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
            context_position_ids=context_rows.view(1, -1),
            context_doc_ids=global_docs.index_select(1, context_rows),
            anchor_doc_ids=global_docs.gather(1, anchors),
            block_size=args.block_size,
            cp_rank=rank,
            cp_size=cp_size,
            group=dist.group.WORLD if cp_size > 1 else None,
            sequence_length=args.context_length,
            backend=args.backend,
        )

        def step() -> None:
            output, _ = direct_cp_attention(*inputs, metadata)
            torch.autograd.grad(output, inputs, dout)

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.runs):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(float(elapsed.item()))

        if rank == 0:
            result = {
                "schema": "aspec.thd-cp-attention-benchmark/v1",
                "environment": {
                    "host": socket.gethostname(),
                    "gpu": torch.cuda.get_device_name(),
                    "world_size": cp_size,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "flash_attn_4": _version("flash-attn-4"),
                    "cudnn_frontend": _version("nvidia-cudnn-frontend"),
                },
                "workload": {
                    "backend": args.backend,
                    "qkv_format": "thd",
                    "packing": True,
                    "context_parallel_size": cp_size,
                    "context_length": args.context_length,
                    "documents": args.documents,
                    "num_anchors": args.num_anchors,
                    "block_size": args.block_size,
                    "query_heads": 32,
                    "kv_heads": 8,
                    "head_dim": 128,
                    "dtype": "bfloat16",
                    "phase": "forward_backward",
                },
                "timing": {
                    "warmup": args.warmup,
                    "runs": args.runs,
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "samples_ms": samples,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
