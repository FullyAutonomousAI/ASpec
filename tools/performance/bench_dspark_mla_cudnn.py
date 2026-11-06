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

"""Benchmark the K3 DSpark materialized-MLA path in cuDNN FlexAttention.

The workload mirrors the tensors at the attention boundary: BSHD, BF16,
``Dqk=192``, ``Dv=128``, and the exact DSpark context-plus-draft interval mask.
It can also include the three transpose materializations used by the initial
adapter, so layout changes are measured separately from kernel-plan changes.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from aspec.models.ops.cudnn_flex_attention import build_dspark_mla_mask


@dataclass(frozen=True)
class Workload:
    context_length: int
    block_size: int
    num_anchors: int
    num_heads: int
    batch_size: int = 1
    qk_head_dim: int = 192
    value_head_dim: int = 128

    @property
    def query_length(self) -> int:
        return self.block_size * self.num_anchors

    @property
    def key_length(self) -> int:
        return self.context_length + self.query_length


def _anchors(workload: Workload) -> torch.Tensor:
    maximum = max(workload.context_length - 1, 0)
    return (
        torch.linspace(
            0,
            maximum,
            workload.num_anchors,
            dtype=torch.float64,
            device="cuda",
        )
        .round()
        .to(torch.long)
        .unsqueeze(0)
    )


def _flush_buffer() -> torch.Tensor:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(properties, "L2_cache_size", 128 * 1024 * 1024))
    return torch.empty(
        max(2 * l2_bytes, 256 * 1024 * 1024),
        dtype=torch.uint8,
        device="cuda",
    )


def _time(fn: Callable[[], Any], flush: torch.Tensor) -> float:
    flush.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _measure(
    fn: Callable[[], Any], *, flush: torch.Tensor, warmup: int, runs: int
) -> dict[str, Any]:
    for _ in range(warmup):
        flush.zero_()
        fn()
        torch.cuda.synchronize()
    samples = [_time(fn, flush) for _ in range(runs)]
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _measure_round_robin(
    functions: dict[str, Callable[[], Any]],
    *,
    flush: torch.Tensor,
    warmup: int,
    runs: int,
) -> dict[str, dict[str, Any]]:
    """Rotate execution order to reduce clock and thermal bias."""

    items = list(functions.items())
    for index in range(warmup):
        offset = index % len(items)
        for _, fn in items[offset:] + items[:offset]:
            flush.zero_()
            fn()
            torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in functions}
    for index in range(runs):
        offset = index % len(items)
        for name, fn in items[offset:] + items[:offset]:
            samples[name].append(_time(fn, flush))
    return {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }


class Runner:
    def __init__(
        self,
        workload: Workload,
        *,
        fwd_variant: str | None,
        legacy_layout_copies: bool,
    ) -> None:
        from cudnn.flex_attention import create_mask_plan, flex_attn_func

        self.workload = workload
        self.legacy_layout_copies = legacy_layout_copies
        shape_q = (
            workload.batch_size,
            workload.query_length,
            workload.num_heads,
            workload.qk_head_dim,
        )
        shape_k = (
            workload.batch_size,
            workload.key_length,
            workload.num_heads,
            workload.qk_head_dim,
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(shape_q, dtype=torch.bfloat16, device="cuda", generator=generator)
        k = torch.randn(shape_k, dtype=torch.bfloat16, device="cuda", generator=generator)
        # kv_b_proj emits [K, H, qk_nope(128) + value(128)].  Its V split is
        # BSHD with a 256-element head stride, so keep that production layout.
        kv_projection = torch.randn(
            shape_k[:-1] + (256,),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        v = kv_projection[..., 128:]
        if legacy_layout_copies:
            self.q = q.transpose(1, 2).contiguous().requires_grad_()
            self.k = k.transpose(1, 2).contiguous().requires_grad_()
            self.v = v.transpose(1, 2).contiguous().requires_grad_()
        else:
            self.q = q.requires_grad_()
            self.k = k.requires_grad_()
            self.v = v.requires_grad_()
        self.dout = torch.randn(
            shape_q[:-1] + (workload.value_head_dim,),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )

        anchors = _anchors(workload)
        mask = build_dspark_mla_mask(
            anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
            context_length=workload.context_length,
            block_size=workload.block_size,
        )
        plan_q, plan_k, plan_v = self._bshd_inputs()
        started = time.perf_counter()
        self.plan = create_mask_plan(
            mask.mask_func,
            plan_q,
            plan_k,
            plan_v,
            pack_gqa=False,
            build_backward=True,
            _fwd_variant=fwd_variant,
        )
        self.plan_build_ms = (time.perf_counter() - started) * 1e3
        self._flex_attn_func = flex_attn_func

    def _bshd_inputs(self):
        if self.legacy_layout_copies:
            return (
                self.q.transpose(1, 2).contiguous(),
                self.k.transpose(1, 2).contiguous(),
                self.v.transpose(1, 2).contiguous(),
            )
        return self.q, self.k, self.v

    def forward(self):
        q, k, v = self._bshd_inputs()
        return self._flex_attn_func(
            q,
            k,
            v,
            mask_plan=self.plan,
            softmax_scale=self.workload.qk_head_dim**-0.5,
        )

    def combined(self):
        output = self.forward()
        return torch.autograd.grad(output, (self.q, self.k, self.v), self.dout)


def _environment() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-lengths", default="4096,32768,131072")
    parser.add_argument("--block-sizes", default="5,7")
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="qstage1_2cta,qstage1_1cta,qstage2_1cta",
    )
    parser.add_argument("--include-legacy-layout", action="store_true")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="warm one workload, execute one combined pass in the profile_combined NVTX range, and exit",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: tuple[str, ...] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    results: dict[str, Any] = {"environment": _environment(), "workloads": []}
    flush = _flush_buffer()
    variants = tuple(item for item in args.variants.split(",") if item)
    context_lengths = _parse_csv_ints(args.context_lengths)
    block_sizes = _parse_csv_ints(args.block_sizes)
    if args.profile_only:
        if len(context_lengths) != 1 or len(block_sizes) != 1 or len(variants) != 1:
            raise ValueError("--profile-only requires one context length, block size, and variant")
        workload = Workload(
            context_length=context_lengths[0],
            block_size=block_sizes[0],
            num_anchors=args.num_anchors,
            num_heads=args.num_heads,
        )
        runner = Runner(
            workload,
            fwd_variant=variants[0],
            legacy_layout_copies=args.include_legacy_layout,
        )
        runner.combined()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push("profile_combined")
        runner.combined()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        return
    for context_length in context_lengths:
        for block_size in block_sizes:
            workload = Workload(
                context_length=context_length,
                block_size=block_size,
                num_anchors=args.num_anchors,
                num_heads=args.num_heads,
            )
            record: dict[str, Any] = {"workload": asdict(workload), "candidates": {}}
            candidates = [(variant, False) for variant in variants]
            if args.include_legacy_layout:
                candidates.append(("qstage1_2cta", True))
            runners: dict[str, Runner] = {}
            for variant, legacy in candidates:
                name = f"{variant}{'_legacy_layout' if legacy else ''}"
                runner = Runner(
                    workload,
                    fwd_variant=variant,
                    legacy_layout_copies=legacy,
                )
                # The first call includes CUTLASS DSL compilation; exclude it.
                runner.combined()
                torch.cuda.synchronize()
                runners[name] = runner
                record["candidates"][name] = {
                    "plan_build_ms": runner.plan_build_ms,
                }
            forward = _measure_round_robin(
                {name: runner.forward for name, runner in runners.items()},
                flush=flush,
                warmup=args.warmup,
                runs=args.runs,
            )
            combined = _measure_round_robin(
                {name: runner.combined for name, runner in runners.items()},
                flush=flush,
                warmup=args.warmup,
                runs=args.runs,
            )
            for name in runners:
                record["candidates"][name]["forward"] = forward[name]
                record["candidates"][name]["combined"] = combined[name]
            results["workloads"].append(record)
            print(json.dumps(record, indent=2), flush=True)
            del runners
            torch.cuda.empty_cache()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
