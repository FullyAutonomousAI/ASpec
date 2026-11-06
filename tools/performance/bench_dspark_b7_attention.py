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

"""Compare DSpark B=5/B=7 FA4 sparse attention with cuDNN-FE flex attention.

The benchmark intentionally uses the kernel-layer DSpark attention shape rather
than the complete model step.  It expects ``direct_cp.py`` and
``fa4_sparse.py`` next to this script when run outside a ASpec checkout.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from aspec.models.ops.direct_cp import DirectCPMetadata, _fa4_partial
    from aspec.models.ops.fa4_sparse import build_direct_cp_sparse_pair
except ModuleNotFoundError:
    from direct_cp import DirectCPMetadata, _fa4_partial
    from fa4_sparse import build_direct_cp_sparse_pair


@dataclass(frozen=True)
class Workload:
    context_length: int
    block_size: int = 7
    num_anchors: int = 512
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16

    @property
    def query_length(self) -> int:
        return self.block_size * self.num_anchors

    @property
    def key_length(self) -> int:
        return self.context_length + self.query_length


@dataclass
class TimingStats:
    samples_ms: list[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "median_ms": statistics.median(self.samples_ms),
            "min_ms": min(self.samples_ms),
            "max_ms": max(self.samples_ms),
            "samples_ms": self.samples_ms,
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(root: str, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-C", root, *args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(cudnn_frontend_root: str) -> dict[str, Any]:
    try:
        driver = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn_4": _package_version("flash-attn-4"),
        "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        "cudnn_frontend": _package_version("nvidia-cudnn-frontend"),
        "cudnn_frontend_commit": _git_output(cudnn_frontend_root, "rev-parse", "HEAD"),
        "cudnn_frontend_dirty": bool(
            _git_output(cudnn_frontend_root, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def make_anchors(workload: Workload, device: torch.device) -> torch.Tensor:
    """Return sorted anchors spread through the valid context range."""

    maximum = max(workload.context_length - workload.block_size, 0)
    anchors = (
        torch.linspace(0, maximum, workload.num_anchors, dtype=torch.float64, device=device)
        .round()
        .to(torch.long)
    )
    return anchors.unsqueeze(0)


def make_endpoints(
    workload: Workload,
    anchors: torch.Tensor,
    mask_shape: str,
) -> torch.Tensor:
    """Encode the DSpark mask or an equal-density ideal prefix mask."""

    query_anchors = anchors.repeat_interleave(workload.block_size, dim=1)
    if mask_shape == "flat_prefix":
        average_visible = int(
            (query_anchors.to(torch.int64) + workload.block_size)
            .sum()
            .div(workload.query_length, rounding_mode="floor")
            .item()
        )
        return torch.full(
            (1, 1, workload.query_length),
            average_visible,
            device=anchors.device,
            dtype=torch.int32,
        )
    if mask_shape != "dspark":
        raise ValueError(f"unsupported mask shape: {mask_shape}")
    block_starts = (
        workload.context_length
        + torch.arange(
            workload.num_anchors, device=anchors.device, dtype=torch.long
        ).repeat_interleave(workload.block_size)
        * workload.block_size
    )
    endpoints = torch.stack(
        (query_anchors[0], block_starts, block_starts + workload.block_size), dim=0
    )
    return endpoints.to(torch.int32).unsqueeze(0).contiguous()


def visible_pairs(workload: Workload, anchors: torch.Tensor) -> int:
    per_block = anchors.to(torch.int64) + workload.block_size
    return int((per_block * workload.block_size).sum().item())


def endpoint_visible_pairs(endpoints: torch.Tensor) -> int:
    counts = endpoints[:, 0].to(torch.int64)
    for index in range(1, endpoints.shape[1], 2):
        counts = counts + endpoints[:, index + 1] - endpoints[:, index]
    return int(counts.sum().item())


def phase_flops(workload: Workload, pairs: int, phase: str) -> int:
    head_pairs = workload.num_q_heads * pairs
    if phase == "forward":
        return 4 * head_pairs * workload.head_dim
    if phase == "backward":
        return 10 * head_pairs * workload.head_dim
    if phase == "combined":
        return 14 * head_pairs * workload.head_dim
    raise ValueError(phase)


def _flush_buffer() -> torch.Tensor:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(properties, "L2_cache_size", 128 * 1024 * 1024))
    return torch.empty(max(2 * l2_bytes, 256 * 1024 * 1024), dtype=torch.uint8, device="cuda")


def _time_cuda(fn: Callable[[], Any], flush: torch.Tensor) -> float:
    flush.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _time_wall(fn: Callable[[], Any], flush: torch.Tensor) -> float:
    flush.zero_()
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def measure(
    fn: Callable[[], Any],
    *,
    flush: torch.Tensor,
    warmup: int,
    runs: int,
    setup: Callable[[], Any] | None = None,
    wall: bool = False,
) -> TimingStats:
    for _ in range(warmup):
        if setup is not None:
            setup()
        flush.zero_()
        fn()
        torch.cuda.synchronize()
    timer = _time_wall if wall else _time_cuda
    samples = []
    for _ in range(runs):
        if setup is not None:
            setup()
        samples.append(timer(fn, flush))
    return TimingStats(samples)


def measure_interleaved_combined(
    baseline: Any,
    candidate: Any,
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    """Alternate execution order to reduce boost/thermal bias between runners."""

    flush = _flush_buffer()
    for index in range(warmup):
        order = (baseline, candidate) if index % 2 == 0 else (candidate, baseline)
        for runner in order:
            flush.zero_()
            runner.combined()
            torch.cuda.synchronize()

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(runs):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        current: dict[str, float] = {}
        for name, runner in order:
            current[name] = _time_cuda(runner.combined, flush)
        baseline_samples.append(current["baseline"])
        candidate_samples.append(current["candidate"])

    speedups = [
        baseline_ms / candidate_ms
        for baseline_ms, candidate_ms in zip(baseline_samples, candidate_samples, strict=True)
    ]
    return {
        "baseline": TimingStats(baseline_samples).to_json(),
        "candidate": TimingStats(candidate_samples).to_json(),
        "paired_speedups": speedups,
        "median_paired_speedup": statistics.median(speedups),
    }


def measure_round_robin_combined(
    runners: dict[str, Any],
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    """Rotate runner order so every candidate sees each thermal position."""

    if not runners:
        raise ValueError("round-robin comparison requires at least one runner")
    items = list(runners.items())
    flush = _flush_buffer()
    for index in range(warmup):
        offset = index % len(items)
        for _, runner in items[offset:] + items[:offset]:
            flush.zero_()
            runner.combined()
            torch.cuda.synchronize()

    samples = {name: [] for name in runners}
    for index in range(runs):
        offset = index % len(items)
        for name, runner in items[offset:] + items[:offset]:
            samples[name].append(_time_cuda(runner.combined, flush))
    return {name: TimingStats(values).to_json() for name, values in samples.items()}


def _block_stats(sparse: Any) -> dict[str, Any]:
    partial = int(sparse.mask_block_cnt.sum().item())
    full = int(sparse.full_block_cnt.sum().item()) if sparse.full_block_cnt is not None else 0
    return {
        "block_size": [int(value) for value in sparse.block_size],
        "partial": partial,
        "full": full,
        "active": partial + full,
    }


class FA4Runner:
    def __init__(
        self, workload: Workload, anchors: torch.Tensor, *, seed: int = 0
    ) -> None:
        self.workload = workload
        q = workload.query_length
        k = workload.key_length
        torch.manual_seed(seed)
        self.query = torch.randn(
            1,
            workload.num_q_heads,
            q,
            workload.head_dim,
            dtype=workload.dtype,
            device="cuda",
            requires_grad=True,
        )
        self.key = torch.randn(
            1,
            workload.num_kv_heads,
            k,
            workload.head_dim,
            dtype=workload.dtype,
            device="cuda",
            requires_grad=True,
        )
        self.value = torch.randn_like(self.key, requires_grad=True)
        self.dout = torch.randn_like(self.query)
        self.metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
            context_position_ids=torch.arange(
                workload.context_length, device="cuda", dtype=torch.long
            ).unsqueeze(0),
            block_size=workload.block_size,
            cp_rank=0,
            cp_size=1,
            group=None,
            backend="fa4",
            fa4_sparse=True,
        )
        block_ids = (
            torch.arange(workload.num_anchors, device="cuda", dtype=torch.int32)
            .repeat_interleave(workload.block_size)
            .unsqueeze(0)
        )
        self._sparse_args = {
            "anchor_positions": anchors.to(torch.int32).contiguous(),
            "query_blocks": block_ids,
            "context_positions": self.metadata.context_position_ids.to(torch.int32),
            "draft_blocks": block_ids,
            "query_token_keep": torch.ones(1, q, device="cuda", dtype=torch.int32),
            "key_token_keep": torch.ones(1, k, device="cuda", dtype=torch.int32),
        }
        self.backward_output: torch.Tensor | None = None

    @property
    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query, self.key, self.value

    def build_metadata(self):
        return build_direct_cp_sparse_pair(**self._sparse_args)

    def forward(self) -> torch.Tensor:
        return _fa4_partial(
            self.query,
            self.key,
            self.value,
            self.metadata,
            self.workload.head_dim**-0.5,
        )[0]

    def prepare_backward(self) -> None:
        self.backward_output = self.forward()

    def backward(self):
        if self.backward_output is None:
            raise RuntimeError("backward graph was not prepared")
        output = self.backward_output
        self.backward_output = None
        return torch.autograd.grad(output, self.qkv, self.dout)

    def combined(self):
        return torch.autograd.grad(self.forward(), self.qkv, self.dout)


class CudnnRunner:
    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        pack_gqa: bool,
        fwd_variant: str | None,
    ) -> None:
        from cudnn.flex_attention import create_mask_plan, flex_attn_func

        self.workload = workload
        self.mask = endpoints
        self.pack_gqa = pack_gqa
        self.fwd_variant = fwd_variant
        self.deterministic = False
        self.query = source.query.detach().permute(0, 2, 1, 3).contiguous().requires_grad_()
        self.key = source.key.detach().permute(0, 2, 1, 3).contiguous().requires_grad_()
        self.value = source.value.detach().permute(0, 2, 1, 3).contiguous().requires_grad_()
        self.dout = source.dout.permute(0, 2, 1, 3).contiguous()
        self._create_mask_plan = create_mask_plan
        self._flex_attn_func = flex_attn_func
        self.plan: Any = None
        self.backward_output: torch.Tensor | None = None

    @property
    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query, self.key, self.value

    def build_metadata(self):
        return self._create_mask_plan(
            self.mask,
            self.query,
            self.key,
            self.value,
            pack_gqa=self.pack_gqa,
            build_backward=True,
            _fwd_variant=self.fwd_variant,
        )

    def forward(self) -> torch.Tensor:
        return self._flex_attn_func(
            self.query,
            self.key,
            self.value,
            mask_plan=self.plan,
            softmax_scale=self.workload.head_dim**-0.5,
            deterministic=self.deterministic,
        )

    def prepare_backward(self) -> None:
        self.backward_output = self.forward()

    def backward(self):
        if self.backward_output is None:
            raise RuntimeError("backward graph was not prepared")
        output = self.backward_output
        self.backward_output = None
        return torch.autograd.grad(output, self.qkv, self.dout)

    def combined(self):
        return torch.autograd.grad(self.forward(), self.qkv, self.dout)

    def fa4_output(self, output: torch.Tensor) -> torch.Tensor:
        return output.permute(0, 2, 1, 3)

    def fa4_grads(
        self, grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(grad.permute(0, 2, 1, 3) for grad in grads)


class CudnnPhysicalGqaRunner(CudnnRunner):
    """Run GQA as equivalent MHA by folding Q heads into the sequence axis."""

    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        fwd_variant: str | None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(workload, endpoints, source, False, fwd_variant)
        self.deterministic = deterministic
        self.q_ratio = workload.num_q_heads // workload.num_kv_heads
        batch = source.query.shape[0]
        seqlen = workload.query_length
        dim = workload.head_dim
        kv_heads = workload.num_kv_heads

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.detach()
                .reshape(batch, kv_heads, self.q_ratio, seqlen, dim)
                .permute(0, 3, 2, 1, 4)
                .contiguous()
                .reshape(batch, seqlen * self.q_ratio, kv_heads, dim)
            )

        self.query = pack(source.query).requires_grad_()
        self.dout = pack(source.dout)
        self.mask = endpoints.repeat_interleave(self.q_ratio, dim=-1).contiguous()

    def _unpack_query(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.shape[0]
        seqlen = self.workload.query_length
        dim = self.workload.head_dim
        kv_heads = self.workload.num_kv_heads
        return (
            tensor.reshape(batch, seqlen, self.q_ratio, kv_heads, dim)
            .permute(0, 3, 2, 1, 4)
            .contiguous()
            .reshape(batch, self.workload.num_q_heads, seqlen, dim)
        )

    def fa4_output(self, output: torch.Tensor) -> torch.Tensor:
        return self._unpack_query(output)

    def fa4_grads(
        self, grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = grads
        return (
            self._unpack_query(dq),
            dk.permute(0, 2, 1, 3),
            dv.permute(0, 2, 1, 3),
        )


class CudnnExplicitPhysicalGqaRunner(CudnnPhysicalGqaRunner):
    """Physical-GQA candidate using reusable preallocated cuDNN-FE APIs."""

    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        fwd_variant: str | None,
    ) -> None:
        super().__init__(workload, endpoints, source, fwd_variant)
        from cudnn.flex_attention import FlexAttentionBwd, FlexAttentionFwd

        self._fwd_api_type = FlexAttentionFwd
        self._bwd_api_type = FlexAttentionBwd
        self.output = torch.empty_like(self.query)
        self.lse = torch.empty(
            (self.query.shape[0], self.query.shape[2], self.query.shape[1]),
            dtype=torch.float32,
            device=self.query.device,
        )
        self.dq = torch.empty_like(self.query)
        self.dk = torch.empty_like(self.key)
        self.dv = torch.empty_like(self.value)
        self.fwd_api: Any = None
        self.bwd_api: Any = None
        self.backward_plan: Any = None
        self.fwd_workspace: torch.Tensor | None = None
        self.bwd_workspace: torch.Tensor | None = None

    def _compile_forward(self) -> None:
        if self.fwd_api is not None:
            return
        self.fwd_api = self._fwd_api_type(
            self.query,
            self.key,
            self.value,
            self.output,
            self.plan,
            self.lse,
        )
        self.fwd_api.check_support()
        self.fwd_api.compile()
        self.fwd_workspace = torch.empty(
            (self.fwd_api.workspace_size,), dtype=torch.uint8, device=self.query.device
        )

    def _compile_backward(self) -> None:
        if self.bwd_api is not None:
            return
        backward_plan = self.backward_plan if self.backward_plan is not None else self.plan
        self.bwd_api = self._bwd_api_type(
            self.query,
            self.key,
            self.value,
            self.output,
            self.dout,
            self.lse,
            self.dq,
            self.dk,
            self.dv,
            backward_plan,
        )
        self.bwd_api.check_support()
        self.bwd_api.compile()
        self.bwd_workspace = torch.empty(
            (self.bwd_api.workspace_size,), dtype=torch.uint8, device=self.query.device
        )

    def forward(self) -> torch.Tensor:
        self._compile_forward()
        self.fwd_api.execute(
            self.query,
            self.key,
            self.value,
            self.output,
            self.plan,
            self.lse,
            workspace=self.fwd_workspace,
            softmax_scale=self.workload.head_dim**-0.5,
        )
        return self.output

    def prepare_backward(self) -> None:
        self.forward()

    def backward(self):
        self._compile_backward()
        backward_plan = self.backward_plan if self.backward_plan is not None else self.plan
        self.bwd_api.execute(
            self.query,
            self.key,
            self.value,
            self.output,
            self.dout,
            self.lse,
            self.dq,
            self.dk,
            self.dv,
            backward_plan,
            workspace=self.bwd_workspace,
            softmax_scale=self.workload.head_dim**-0.5,
        )
        return self.dq, self.dk, self.dv

    def combined(self):
        self.forward()
        return self.backward()


class CudnnGraphPhysicalGqaRunner(CudnnExplicitPhysicalGqaRunner):
    """Physical-GQA candidate with reusable CUDA graphs for fixed-shape training."""

    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        fwd_variant: str | None,
    ) -> None:
        super().__init__(workload, endpoints, source, fwd_variant)
        self.backward_graph: torch.cuda.CUDAGraph | None = None
        self.combined_graph: torch.cuda.CUDAGraph | None = None

    @property
    def _grads(self):
        return self.dq, self.dk, self.dv

    def _capture_backward(self) -> None:
        CudnnExplicitPhysicalGqaRunner.backward(self)
        torch.cuda.synchronize()
        self.backward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.backward_graph):
            CudnnExplicitPhysicalGqaRunner.backward(self)

    def backward(self):
        if self.backward_graph is None:
            self._capture_backward()
        self.backward_graph.replay()
        return self._grads

    def _capture_combined(self) -> None:
        CudnnExplicitPhysicalGqaRunner.forward(self)
        CudnnExplicitPhysicalGqaRunner.backward(self)
        torch.cuda.synchronize()
        self.combined_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.combined_graph):
            CudnnExplicitPhysicalGqaRunner.forward(self)
            CudnnExplicitPhysicalGqaRunner.backward(self)

    def combined(self):
        if self.combined_graph is None:
            self._capture_combined()
        self.combined_graph.replay()
        return self._grads


class CudnnDualPlanPhysicalGqaRunner(CudnnExplicitPhysicalGqaRunner):
    """Use the fastest independently selected plan for forward and backward."""

    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        backward_variant: str,
        forward_variant: str | None = None,
    ) -> None:
        super().__init__(workload, endpoints, source, forward_variant)
        self.backward_variant = backward_variant

    def build_metadata(self):
        forward_plan = super().build_metadata()
        self.backward_plan = self._create_mask_plan(
            self.mask,
            self.query,
            self.key,
            self.value,
            pack_gqa=False,
            build_backward=True,
            _fwd_variant=self.backward_variant,
        )
        return forward_plan


class CudnnGraphDualPlanPhysicalGqaRunner(CudnnDualPlanPhysicalGqaRunner):
    """Dual-plan candidate with graph-captured fixed-shape forward and backward."""

    def __init__(
        self,
        workload: Workload,
        endpoints: torch.Tensor,
        source: FA4Runner,
        backward_variant: str,
        forward_variant: str | None = None,
    ) -> None:
        super().__init__(
            workload,
            endpoints,
            source,
            backward_variant,
            forward_variant,
        )
        self.combined_graph: torch.cuda.CUDAGraph | None = None

    def _capture_combined(self) -> None:
        CudnnExplicitPhysicalGqaRunner.forward(self)
        CudnnExplicitPhysicalGqaRunner.backward(self)
        torch.cuda.synchronize()
        self.combined_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.combined_graph):
            CudnnExplicitPhysicalGqaRunner.forward(self)
            CudnnExplicitPhysicalGqaRunner.backward(self)

    def combined(self):
        if self.combined_graph is None:
            self._capture_combined()
        self.combined_graph.replay()
        return self.dq, self.dk, self.dv


class CudnnB7SplitRunner(CudnnRunner):
    """Two-plan B=7 prototype: dense common prefix plus a short ragged tail."""

    def __init__(
        self,
        workload: Workload,
        anchors: torch.Tensor,
        source: FA4Runner,
        pack_gqa: bool,
        fwd_variant: str | None,
        residual_fwd_variant: str | None,
        group_queries: int,
    ) -> None:
        if group_queries <= 0:
            raise ValueError("group_queries must be positive")
        query_anchors = anchors.repeat_interleave(workload.block_size, dim=1)[0]
        query_length = workload.query_length
        padded_length = ((query_length + group_queries - 1) // group_queries) * group_queries
        padded = torch.nn.functional.pad(
            query_anchors,
            (0, padded_length - query_length),
            value=workload.context_length,
        )
        common = (
            padded.view(-1, group_queries)
            .amin(dim=1)
            .repeat_interleave(group_queries)[:query_length]
        )
        common_mask = common.to(torch.int32).view(1, 1, query_length).contiguous()
        block_starts = (
            workload.context_length
            + torch.arange(
                workload.num_anchors,
                device=anchors.device,
                dtype=torch.long,
            ).repeat_interleave(workload.block_size)
            * workload.block_size
        )
        residual_mask = torch.stack(
            (
                torch.zeros_like(common),
                common,
                query_anchors,
                block_starts,
                block_starts + workload.block_size,
            ),
            dim=0,
        ).to(torch.int32)
        residual_mask = residual_mask.unsqueeze(0).contiguous()
        super().__init__(
            workload,
            common_mask,
            source,
            pack_gqa,
            fwd_variant,
        )
        self.residual_mask = residual_mask
        self.residual_fwd_variant = residual_fwd_variant
        self.group_queries = group_queries

    def _build_one_plan(self, mask: torch.Tensor, fwd_variant: str | None):
        return self._create_mask_plan(
            mask,
            self.query,
            self.key,
            self.value,
            pack_gqa=self.pack_gqa,
            build_backward=True,
            _fwd_variant=fwd_variant,
        )

    def build_metadata(self):
        return self._build_one_plan(self.mask, self.fwd_variant), self._build_one_plan(
            self.residual_mask, self.residual_fwd_variant
        )

    def forward(self) -> torch.Tensor:
        common_plan, residual_plan = self.plan
        common_out, common_lse = self._flex_attn_func(
            self.query,
            self.key,
            self.value,
            mask_plan=common_plan,
            softmax_scale=self.workload.head_dim**-0.5,
            deterministic=False,
            return_lse=True,
        )
        residual_out, residual_lse = self._flex_attn_func(
            self.query,
            self.key,
            self.value,
            mask_plan=residual_plan,
            softmax_scale=self.workload.head_dim**-0.5,
            deterministic=False,
            return_lse=True,
        )
        if common_lse.requires_grad:
            common_lse.register_hook(lambda grad: grad.contiguous())
            residual_lse.register_hook(lambda grad: grad.contiguous())
        merged_lse = torch.logaddexp(common_lse, residual_lse)
        common_weight = (
            torch.where(
                torch.isfinite(common_lse),
                torch.exp(common_lse - merged_lse),
                torch.zeros_like(common_lse),
            )
            .permute(0, 2, 1)
            .unsqueeze(-1)
        )
        residual_weight = torch.exp(residual_lse - merged_lse).permute(0, 2, 1).unsqueeze(-1)
        return torch.nan_to_num(common_out) * common_weight + residual_out * residual_weight


def _sum_block_stats(plans: tuple[Any, ...]) -> dict[str, Any]:
    stats = [_block_stats(plan._runtime_args[0]) for plan in plans]
    return {
        "block_size": stats[0]["block_size"],
        "partial": sum(stat["partial"] for stat in stats),
        "full": sum(stat["full"] for stat in stats),
        "active": sum(stat["active"] for stat in stats),
        "plans": stats,
    }


def prepare_runner(runner: Any) -> dict[str, Any]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    runner.plan = runner.build_metadata()
    torch.cuda.synchronize()
    metadata_compile_ms = (time.perf_counter() - started) * 1e3
    if isinstance(runner, FA4Runner):
        runner.metadata._fa4_cache.clear()
        runner.metadata._fa4_cache[
            (runner.query.device, runner.workload.query_length, runner.workload.key_length)
        ] = runner.plan
        forward_sparse = runner.plan.forward
        backward_sparse = runner.plan.backward
    elif isinstance(runner, CudnnB7SplitRunner):
        forward_sparse = None
        backward_sparse = None
    else:
        packed = runner.plan._runtime_args[0]
        forward_sparse = packed
        backward_sparse = packed

    started = time.perf_counter()
    runner.forward()
    torch.cuda.synchronize()
    forward_compile_ms = (time.perf_counter() - started) * 1e3
    runner.prepare_backward()
    torch.cuda.synchronize()
    started = time.perf_counter()
    runner.backward()
    torch.cuda.synchronize()
    backward_compile_ms = (time.perf_counter() - started) * 1e3
    if isinstance(runner, CudnnB7SplitRunner):
        forward_stats = _sum_block_stats(runner.plan)
        backward_stats = forward_stats
    else:
        forward_stats = _block_stats(forward_sparse)
        backward_stats = _block_stats(backward_sparse)
    return {
        "compile_ms": {
            "metadata": metadata_compile_ms,
            "forward": forward_compile_ms,
            "backward": backward_compile_ms,
        },
        "forward_blocks": forward_stats,
        "backward_blocks": backward_stats,
    }


def benchmark_runner(
    runner: Any,
    *,
    pairs: int,
    warmup: int,
    runs: int,
    metadata_runs: int,
) -> dict[str, Any]:
    prepared = prepare_runner(runner)
    flush = _flush_buffer()
    metrics: dict[str, Any] = {}
    actions = (
        ("metadata", runner.build_metadata, None, True, metadata_runs),
        ("forward", runner.forward, None, False, runs),
        ("backward", runner.backward, runner.prepare_backward, False, runs),
        ("combined", runner.combined, None, False, runs),
    )
    for name, action, setup, wall, count in actions:
        stats = measure(
            action,
            flush=flush,
            warmup=0 if name == "metadata" else warmup,
            runs=count,
            setup=setup,
            wall=wall,
        ).to_json()
        if name != "metadata":
            stats["active_tflops"] = phase_flops(runner.workload, pairs, name) / (
                stats["median_ms"] * 1e9
            )
        metrics[name] = stats
    del flush
    return {**prepared, "metrics": metrics}


def correctness(fa4: FA4Runner, cudnn: CudnnRunner) -> dict[str, float]:
    fa4_out = fa4.forward()
    cudnn_out = cudnn.fa4_output(cudnn.forward())
    fa4_grads = torch.autograd.grad(fa4_out, fa4.qkv, fa4.dout)
    if isinstance(cudnn, CudnnExplicitPhysicalGqaRunner):
        cudnn_grads = cudnn.fa4_grads(cudnn.backward())
    else:
        cudnn_grads = cudnn.fa4_grads(torch.autograd.grad(cudnn_out, cudnn.qkv, fa4.dout))
    errors: dict[str, float] = {}
    for name, lhs, rhs in (
        ("output", fa4_out, cudnn_out),
        ("dq", fa4_grads[0], cudnn_grads[0]),
        ("dk", fa4_grads[1], cudnn_grads[1]),
        ("dv", fa4_grads[2], cudnn_grads[2]),
    ):
        delta = (lhs.float() - rhs.float()).abs()
        errors[f"{name}_max_abs"] = delta.max().item()
        errors[f"{name}_mean_abs"] = delta.mean().item()
    return errors


def run_case(
    workload: Workload,
    *,
    benchmark_fa4: bool,
    pack_gqa: bool,
    cudnn_fwd_variant: str | None,
    mask_shape: str,
    b7_split_group_queries: int,
    b7_residual_fwd_variant: str | None,
    physical_gqa: bool,
    physical_gqa_deterministic: bool,
    physical_gqa_explicit: bool,
    physical_gqa_graph: bool,
    physical_gqa_dual_plan: bool,
    physical_gqa_dual_plan_graph: bool,
    physical_gqa_forward_variant: str | None,
    physical_gqa_backward_variant: str,
    sweep_physical_plan_variants: bool,
    interleaved_runs: int,
    warmup: int,
    runs: int,
    metadata_runs: int,
    check_correctness: bool,
) -> dict[str, Any]:
    anchors = make_anchors(workload, torch.device("cuda"))
    endpoints = make_endpoints(workload, anchors, mask_shape)
    pairs = endpoint_visible_pairs(endpoints)
    fa4 = FA4Runner(workload, anchors)
    cudnn = CudnnRunner(
        workload,
        endpoints,
        fa4,
        pack_gqa,
        cudnn_fwd_variant,
    )
    candidate_runner: Any = None

    print(
        f"context={workload.context_length:,} q={workload.query_length:,} "
        f"k={workload.key_length:,} density={pairs / (workload.query_length * workload.key_length):.3%}",
        flush=True,
    )
    fa4_result = (
        benchmark_runner(fa4, pairs=pairs, warmup=warmup, runs=runs, metadata_runs=metadata_runs)
        if benchmark_fa4
        else None
    )
    cudnn_result = benchmark_runner(
        cudnn, pairs=pairs, warmup=warmup, runs=runs, metadata_runs=metadata_runs
    )
    split_result = None
    if b7_split_group_queries:
        split = CudnnB7SplitRunner(
            workload,
            anchors,
            fa4,
            pack_gqa,
            cudnn_fwd_variant,
            b7_residual_fwd_variant,
            b7_split_group_queries,
        )
        split_result = benchmark_runner(
            split,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        if check_correctness:
            split_result["correctness"] = correctness(fa4, split)
    physical_gqa_result = None
    if physical_gqa:
        physical = CudnnPhysicalGqaRunner(
            workload,
            endpoints,
            fa4,
            cudnn_fwd_variant,
            physical_gqa_deterministic,
        )
        physical_gqa_result = benchmark_runner(
            physical,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        candidate_runner = physical
        if check_correctness:
            physical_gqa_result["correctness"] = correctness(fa4, physical)
    physical_gqa_explicit_result = None
    if physical_gqa_explicit:
        physical_explicit = CudnnExplicitPhysicalGqaRunner(
            workload,
            endpoints,
            fa4,
            cudnn_fwd_variant,
        )
        physical_gqa_explicit_result = benchmark_runner(
            physical_explicit,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        candidate_runner = physical_explicit
        if check_correctness:
            physical_gqa_explicit_result["correctness"] = correctness(fa4, physical_explicit)
    physical_gqa_graph_result = None
    if physical_gqa_graph:
        physical_graph = CudnnGraphPhysicalGqaRunner(
            workload,
            endpoints,
            fa4,
            cudnn_fwd_variant,
        )
        physical_gqa_graph_result = benchmark_runner(
            physical_graph,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        candidate_runner = physical_graph
        if check_correctness:
            physical_gqa_graph_result["correctness"] = correctness(fa4, physical_graph)
    physical_gqa_dual_plan_result = None
    if physical_gqa_dual_plan:
        physical_dual = CudnnDualPlanPhysicalGqaRunner(
            workload,
            endpoints,
            fa4,
            physical_gqa_backward_variant,
            physical_gqa_forward_variant,
        )
        physical_gqa_dual_plan_result = benchmark_runner(
            physical_dual,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        candidate_runner = physical_dual
        if check_correctness:
            physical_gqa_dual_plan_result["correctness"] = correctness(fa4, physical_dual)
    physical_gqa_dual_plan_graph_result = None
    if physical_gqa_dual_plan_graph:
        physical_dual_graph = CudnnGraphDualPlanPhysicalGqaRunner(
            workload,
            endpoints,
            fa4,
            physical_gqa_backward_variant,
            physical_gqa_forward_variant,
        )
        physical_gqa_dual_plan_graph_result = benchmark_runner(
            physical_dual_graph,
            pairs=pairs,
            warmup=warmup,
            runs=runs,
            metadata_runs=metadata_runs,
        )
        candidate_runner = physical_dual_graph
        if check_correctness:
            physical_gqa_dual_plan_graph_result["correctness"] = correctness(
                fa4, physical_dual_graph
            )
    physical_plan_sweep = None
    if sweep_physical_plan_variants:
        variants = (None, "qstage2_1cta", "qstage1_1cta", "qstage1_2cta")
        sweep_runners: dict[str, Any] = {"stock": cudnn}
        for forward_variant in variants:
            forward_name = forward_variant or "default"
            for backward_variant in variants[1:]:
                name = f"fwd={forward_name},bwd={backward_variant}"
                runner = CudnnGraphDualPlanPhysicalGqaRunner(
                    workload,
                    endpoints,
                    fa4,
                    backward_variant,
                    forward_variant,
                )
                prepare_runner(runner)
                sweep_runners[name] = runner
        physical_plan_sweep = measure_round_robin_combined(
            sweep_runners,
            warmup=warmup,
            runs=interleaved_runs or runs,
        )
        stock_ms = physical_plan_sweep["stock"]["median_ms"]
        best_name, best_stats = min(
            ((name, stats) for name, stats in physical_plan_sweep.items() if name != "stock"),
            key=lambda item: item[1]["median_ms"],
        )
        print(
            "  physical plan sweep "
            f"stock={stock_ms:.3f}ms best={best_name} "
            f"{best_stats['median_ms']:.3f}ms "
            f"speedup={stock_ms / best_stats['median_ms']:.3f}x",
            flush=True,
        )
    interleaved = None
    if interleaved_runs:
        if candidate_runner is None:
            raise ValueError("interleaved comparison requires a physical-GQA candidate")
        interleaved = measure_interleaved_combined(
            cudnn,
            candidate_runner,
            warmup=warmup,
            runs=interleaved_runs,
        )
        print(
            "  combined interleaved "
            f"cuDNN={interleaved['baseline']['median_ms']:.3f}ms "
            f"candidate={interleaved['candidate']['median_ms']:.3f}ms "
            f"paired_speedup={interleaved['median_paired_speedup']:.3f}x",
            flush=True,
        )
    checks = correctness(fa4, cudnn) if check_correctness else None
    for phase in ("forward", "backward", "combined"):
        cudnn_ms = cudnn_result["metrics"][phase]["median_ms"]
        if fa4_result is not None:
            fa4_ms = fa4_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} FA4={fa4_ms:.3f}ms cuDNN={cudnn_ms:.3f}ms "
                f"FA4_speedup={cudnn_ms / fa4_ms:.3f}x",
                flush=True,
            )
        else:
            print(f"  {phase:8s} cuDNN={cudnn_ms:.3f}ms", flush=True)
        if split_result is not None:
            split_ms = split_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} B7-split={split_ms:.3f}ms vs_cuDNN={cudnn_ms / split_ms:.3f}x",
                flush=True,
            )
        if physical_gqa_result is not None:
            physical_ms = physical_gqa_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} physical-GQA={physical_ms:.3f}ms "
                f"vs_cuDNN={cudnn_ms / physical_ms:.3f}x",
                flush=True,
            )
        if physical_gqa_explicit_result is not None:
            explicit_ms = physical_gqa_explicit_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} physical-GQA-explicit={explicit_ms:.3f}ms "
                f"vs_cuDNN={cudnn_ms / explicit_ms:.3f}x",
                flush=True,
            )
        if physical_gqa_graph_result is not None:
            graph_ms = physical_gqa_graph_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} physical-GQA-graph={graph_ms:.3f}ms "
                f"vs_cuDNN={cudnn_ms / graph_ms:.3f}x",
                flush=True,
            )
        if physical_gqa_dual_plan_result is not None:
            dual_ms = physical_gqa_dual_plan_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} physical-GQA-dual-plan={dual_ms:.3f}ms "
                f"vs_cuDNN={cudnn_ms / dual_ms:.3f}x",
                flush=True,
            )
        if physical_gqa_dual_plan_graph_result is not None:
            dual_graph_ms = physical_gqa_dual_plan_graph_result["metrics"][phase]["median_ms"]
            print(
                f"  {phase:8s} physical-GQA-dual-plan-graph={dual_graph_ms:.3f}ms "
                f"vs_cuDNN={cudnn_ms / dual_graph_ms:.3f}x",
                flush=True,
            )
    del cudnn, fa4
    torch.cuda.empty_cache()
    return {
        "workload": {
            **asdict(workload),
            "dtype": str(workload.dtype).removeprefix("torch."),
            "query_length": workload.query_length,
            "key_length": workload.key_length,
            "anchor_distribution": "sorted_uniform_in_[0,context-block_size]",
        },
        "visible_pairs": pairs,
        "element_density": pairs / (workload.query_length * workload.key_length),
        "pack_gqa": pack_gqa,
        "cudnn_fwd_variant": cudnn_fwd_variant,
        "mask_shape": mask_shape,
        "b7_split_group_queries": b7_split_group_queries,
        "b7_residual_fwd_variant": b7_residual_fwd_variant,
        "correctness": checks,
        "fa4": fa4_result,
        "cudnn_fe": cudnn_result,
        "cudnn_b7_split": split_result,
        "cudnn_physical_gqa": physical_gqa_result,
        "cudnn_physical_gqa_explicit": physical_gqa_explicit_result,
        "cudnn_physical_gqa_graph": physical_gqa_graph_result,
        "cudnn_physical_gqa_dual_plan": physical_gqa_dual_plan_result,
        "cudnn_physical_gqa_dual_plan_graph": physical_gqa_dual_plan_graph_result,
        "physical_plan_sweep": physical_plan_sweep,
        "interleaved_combined": interleaved,
    }


def profile_case(
    workload: Workload,
    *,
    pack_gqa: bool,
    cudnn_fwd_variant: str | None,
    mask_shape: str,
    phase: str,
    physical_gqa: bool,
    physical_gqa_deterministic: bool,
) -> None:
    """Profile one warmed-up cuDNN-FE call without JIT or mask-plan work."""

    anchors = make_anchors(workload, torch.device("cuda"))
    endpoints = make_endpoints(workload, anchors, mask_shape)
    source = FA4Runner(workload, anchors)
    runner = (
        CudnnPhysicalGqaRunner(
            workload,
            endpoints,
            source,
            cudnn_fwd_variant,
            physical_gqa_deterministic,
        )
        if physical_gqa
        else CudnnRunner(
            workload,
            endpoints,
            source,
            pack_gqa,
            cudnn_fwd_variant,
        )
    )
    prepare_runner(runner)
    if phase == "forward":
        action = runner.forward
    elif phase == "backward":
        runner.prepare_backward()
        torch.cuda.synchronize()
        action = runner.backward
    elif phase == "combined":
        action = runner.combined
    else:
        raise ValueError(f"unsupported profile phase: {phase}")

    range_name = f"cudnn_flex_b{workload.block_size}_{phase}"
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range(range_name):
        action()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"profiled {range_name}: context={workload.context_length} "
        f"q={workload.query_length} k={workload.key_length}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=(4096, 131072, 262144))
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--pack-gqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cudnn-fwd-variant",
        choices=("qstage2_1cta", "qstage1_1cta", "qstage1_2cta"),
        default=None,
    )
    parser.add_argument("--mask-shape", choices=("dspark", "flat_prefix"), default="dspark")
    parser.add_argument("--b7-split-group-queries", type=int, default=0)
    parser.add_argument("--physical-gqa", action="store_true")
    parser.add_argument("--physical-gqa-deterministic", action="store_true")
    parser.add_argument("--physical-gqa-explicit", action="store_true")
    parser.add_argument("--physical-gqa-graph", action="store_true")
    parser.add_argument("--physical-gqa-dual-plan", action="store_true")
    parser.add_argument("--physical-gqa-dual-plan-graph", action="store_true")
    parser.add_argument(
        "--physical-gqa-forward-variant",
        choices=("qstage2_1cta", "qstage1_1cta", "qstage1_2cta"),
        default=None,
    )
    parser.add_argument(
        "--physical-gqa-backward-variant",
        choices=("qstage2_1cta", "qstage1_1cta", "qstage1_2cta"),
        default="qstage2_1cta",
    )
    parser.add_argument("--sweep-physical-plan-variants", action="store_true")
    parser.add_argument("--interleaved-runs", type=int, default=0)
    parser.add_argument(
        "--b7-residual-fwd-variant",
        choices=("qstage2_1cta", "qstage1_1cta", "qstage1_2cta"),
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--metadata-runs", type=int, default=3)
    parser.add_argument(
        "--skip-fa4-benchmark",
        action="store_true",
        help="skip FA4 timing while retaining its tensors as cuDNN inputs",
    )
    parser.add_argument("--correctness-context", type=int, default=4096)
    parser.add_argument(
        "--profile-phase",
        choices=("forward", "backward", "combined"),
        default=None,
        help="profile one warmed-up cuDNN-FE call and skip benchmarking",
    )
    parser.add_argument("--cudnn-frontend-root", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.profile_phase is not None:
        if len(args.contexts) != 1:
            raise ValueError("profile mode requires exactly one context")
        profile_case(
            Workload(
                context_length=args.contexts[0],
                block_size=args.block_size,
                num_anchors=args.anchors,
                num_q_heads=args.q_heads,
                num_kv_heads=args.kv_heads,
                head_dim=args.head_dim,
            ),
            pack_gqa=args.pack_gqa,
            cudnn_fwd_variant=args.cudnn_fwd_variant,
            mask_shape=args.mask_shape,
            phase=args.profile_phase,
            physical_gqa=args.physical_gqa,
            physical_gqa_deterministic=args.physical_gqa_deterministic,
        )
        return
    if args.output is None:
        raise ValueError("--output is required outside profile mode")
    results = {
        "environment": collect_environment(args.cudnn_frontend_root),
        "measurement": {
            "cold_l2": True,
            "warmup": args.warmup,
            "runs": args.runs,
            "metadata_runs": args.metadata_runs,
            "timing": "CUDA events for kernels; synchronized wall clock for metadata",
        },
        "cases": [],
    }
    for context in args.contexts:
        workload = Workload(
            context_length=context,
            block_size=args.block_size,
            num_anchors=args.anchors,
            num_q_heads=args.q_heads,
            num_kv_heads=args.kv_heads,
            head_dim=args.head_dim,
        )
        results["cases"].append(
            run_case(
                workload,
                benchmark_fa4=not args.skip_fa4_benchmark,
                pack_gqa=args.pack_gqa,
                cudnn_fwd_variant=args.cudnn_fwd_variant,
                mask_shape=args.mask_shape,
                b7_split_group_queries=args.b7_split_group_queries,
                b7_residual_fwd_variant=args.b7_residual_fwd_variant,
                physical_gqa=args.physical_gqa,
                physical_gqa_deterministic=args.physical_gqa_deterministic,
                physical_gqa_explicit=args.physical_gqa_explicit,
                physical_gqa_graph=args.physical_gqa_graph,
                physical_gqa_dual_plan=args.physical_gqa_dual_plan,
                physical_gqa_dual_plan_graph=args.physical_gqa_dual_plan_graph,
                physical_gqa_forward_variant=args.physical_gqa_forward_variant,
                physical_gqa_backward_variant=args.physical_gqa_backward_variant,
                sweep_physical_plan_variants=args.sweep_physical_plan_variants,
                interleaved_runs=args.interleaved_runs,
                warmup=args.warmup,
                runs=args.runs,
                metadata_runs=args.metadata_runs,
                check_correctness=context == args.correctness_context,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
