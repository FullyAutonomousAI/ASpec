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

"""Validate cuDNN-FE DSpark GQA against ASpec FlexAttention and FA4.

The short-context matrix compares all three implementations.  Long-context
cases compare cuDNN-FE with FA4 because that is ASpec's production backend
for the corresponding route.  Every comparison covers output and dQ/dK/dV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)
from tools.performance.bench_dspark_b7_attention import (
    CudnnGraphDualPlanPhysicalGqaRunner,
    CudnnRunner,
    FA4Runner,
    Workload,
    collect_environment,
    make_endpoints,
)


def make_anchors(
    workload: Workload, *, pattern: str, seed: int, device: torch.device
) -> torch.Tensor:
    maximum = workload.context_length - 1
    if pattern == "uniform":
        anchors = torch.linspace(
            0, maximum, workload.num_anchors, dtype=torch.float64, device=device
        ).round()
    elif pattern == "random":
        generator = torch.Generator(device=device).manual_seed(seed + 1009)
        anchors = torch.randperm(maximum + 1, generator=generator, device=device)[
            : workload.num_anchors
        ].sort().values
    elif pattern == "edge":
        lower_count = workload.num_anchors // 2
        upper_count = workload.num_anchors - lower_count
        width = min(maximum + 1, max(workload.block_size * 8, workload.num_anchors))
        lower = torch.linspace(0, width - 1, lower_count, device=device)
        upper = torch.linspace(maximum - width + 1, maximum, upper_count, device=device)
        anchors = torch.cat((lower, upper)).round()
    else:
        raise ValueError(f"unsupported anchor pattern: {pattern}")
    return anchors.to(torch.long).clamp_(0, maximum).sort().values.unsqueeze(0)


class ASpecFlexRunner:
    """ASpec's built-in PyTorch FlexAttention path on shared FA4 tensors."""

    def __init__(
        self, workload: Workload, anchors: torch.Tensor, source: FA4Runner
    ) -> None:
        self.workload = workload
        self.query = source.query.detach().clone().requires_grad_()
        self.key = source.key.detach().clone().requires_grad_()
        self.value = source.value.detach().clone().requires_grad_()
        self.dout = source.dout.detach().clone()
        keep = torch.ones_like(anchors, dtype=torch.bool)
        mask_mod = _create_dflash_mask_mod(
            anchor_positions=anchors,
            block_keep_mask=keep,
            ctx_len=workload.context_length,
            block_size=workload.block_size,
        )
        self.block_mask = compile_friendly_create_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=workload.query_length,
            KV_LEN=workload.key_length,
            device=torch.device("cuda"),
        )

    @property
    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query, self.key, self.value

    def output_and_grads(self):
        output = compile_friendly_flex_attention(
            query=self.query,
            key=self.key,
            value=self.value,
            block_mask=self.block_mask,
            enable_gqa=True,
            scale=self.workload.head_dim**-0.5,
        )
        grads = torch.autograd.grad(output, self.qkv, self.dout)
        return output, grads


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    ref = reference.float()
    got = actual.float()
    delta = got - ref
    ref_norm = torch.linalg.vector_norm(ref)
    delta_norm = torch.linalg.vector_norm(delta)
    denominator = max(float(ref_norm.item()), torch.finfo(torch.float32).tiny)
    return {
        "shape": list(reference.shape),
        "all_finite": bool(torch.isfinite(got).all()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms_abs": float(delta.square().mean().sqrt().item()),
        "rel_l2": float(delta_norm.item()) / denominator,
    }


def _compare(
    reference: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    actual: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> dict[str, dict[str, Any]]:
    reference_output, reference_grads = reference
    actual_output, actual_grads = actual
    return {
        name: _metrics(lhs, rhs)
        for name, lhs, rhs in (
            ("output", reference_output, actual_output),
            ("dq", reference_grads[0], actual_grads[0]),
            ("dk", reference_grads[1], actual_grads[1]),
            ("dv", reference_grads[2], actual_grads[2]),
        )
    }


def _assert_acceptable(comparisons: dict[str, dict[str, dict[str, Any]]]) -> None:
    failures = []
    for pair, tensors in comparisons.items():
        for name, metrics in tensors.items():
            if (
                not metrics["all_finite"]
                or metrics["max_abs"] > 0.0625
                or metrics["rel_l2"] > 0.02
            ):
                failures.append((pair, name, metrics))
    if failures:
        raise AssertionError(f"accuracy threshold exceeded: {failures}")


def validate_case(
    *,
    context_length: int,
    block_size: int,
    num_anchors: int,
    pattern: str,
    seed: int,
    include_flex: bool,
) -> dict[str, Any]:
    workload = Workload(
        context_length=context_length,
        block_size=block_size,
        num_anchors=num_anchors,
    )
    anchors = make_anchors(
        workload, pattern=pattern, seed=seed, device=torch.device("cuda")
    )
    source = FA4Runner(workload, anchors, seed=seed)
    endpoints = make_endpoints(workload, anchors, "dspark")

    fa4_output = source.forward()
    fa4 = (fa4_output, torch.autograd.grad(fa4_output, source.qkv, source.dout))

    stock = CudnnRunner(workload, endpoints, source, True, None)
    stock.plan = stock.build_metadata()
    stock_output_native = stock.forward()
    stock_output = stock.fa4_output(stock_output_native)
    stock_grads = stock.fa4_grads(
        torch.autograd.grad(stock_output_native, stock.qkv, stock.dout)
    )
    stock_result = (stock_output, stock_grads)

    physical = CudnnGraphDualPlanPhysicalGqaRunner(
        workload,
        endpoints,
        source,
        backward_variant="qstage1_1cta",
        forward_variant=None,
    )
    physical.plan = physical.build_metadata()
    physical.combined()
    physical_result = (
        physical.fa4_output(physical.output),
        physical.fa4_grads((physical.dq, physical.dk, physical.dv)),
    )

    implementations = {
        "fa4": fa4,
        "cudnn_stock": stock_result,
        "cudnn_physical_gqa_graph": physical_result,
    }
    if include_flex:
        implementations["aspec_flex_attention"] = ASpecFlexRunner(
            workload, anchors, source
        ).output_and_grads()

    reference_name = "aspec_flex_attention" if include_flex else "fa4"
    comparisons = {
        f"{reference_name}_vs_{name}": _compare(implementations[reference_name], result)
        for name, result in implementations.items()
        if name != reference_name
    }
    if include_flex:
        comparisons["fa4_vs_cudnn_physical_gqa_graph"] = _compare(
            fa4, physical_result
        )
    _assert_acceptable(comparisons)
    del source, stock, physical, implementations
    torch.cuda.empty_cache()
    return {
        "context_length": context_length,
        "block_size": block_size,
        "num_anchors": num_anchors,
        "query_length": workload.query_length,
        "key_length": workload.key_length,
        "anchor_pattern": pattern,
        "seed": seed,
        "reference": reference_name,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-context", type=int, default=4096)
    parser.add_argument("--long-contexts", type=int, nargs="*", default=(32768, 131072))
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(5, 7))
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--patterns", nargs="+", default=("uniform", "random", "edge"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 17))
    parser.add_argument("--long-patterns", nargs="+", default=("uniform", "random"))
    parser.add_argument("--long-seeds", type=int, nargs="+", default=(0, 17))
    parser.add_argument("--cudnn-frontend-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.anchors > args.short_context:
        raise ValueError("anchors must not exceed the short context length")

    cases = []
    for block_size in args.block_sizes:
        for pattern in args.patterns:
            for seed in args.seeds:
                case = validate_case(
                    context_length=args.short_context,
                    block_size=block_size,
                    num_anchors=args.anchors,
                    pattern=pattern,
                    seed=seed,
                    include_flex=True,
                )
                cases.append(case)
                print(
                    f"PASS short B={block_size} pattern={pattern} seed={seed}",
                    flush=True,
                )
        for context_length in args.long_contexts:
            for pattern in args.long_patterns:
                for seed in args.long_seeds:
                    case = validate_case(
                        context_length=context_length,
                        block_size=block_size,
                        num_anchors=args.anchors,
                        pattern=pattern,
                        seed=seed,
                        include_flex=False,
                    )
                    cases.append(case)
                    print(
                        f"PASS long ctx={context_length} B={block_size} "
                        f"pattern={pattern} seed={seed}",
                        flush=True,
                    )

    maxima: dict[str, dict[str, float]] = {}
    for case in cases:
        for pair, tensors in case["comparisons"].items():
            pair_maxima = maxima.setdefault(pair, {})
            for tensor_name, metrics in tensors.items():
                for metric_name in ("max_abs", "mean_abs", "rms_abs", "rel_l2"):
                    key = f"{tensor_name}_{metric_name}"
                    pair_maxima[key] = max(pair_maxima.get(key, 0.0), metrics[metric_name])

    result = {
        "environment": collect_environment(args.cudnn_frontend_root),
        "acceptance": {"max_abs": 0.0625, "rel_l2": 0.02, "all_finite": True},
        "case_count": len(cases),
        "maxima": maxima,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
