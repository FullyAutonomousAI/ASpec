# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Watermark-buffered, backpressure-aware batch dispatcher for inference engines.

Sits between the training controller and Ray inference engine actors.
Refills its prompt buffer when it drops below a low-watermark, dispatches
fixed-size batches to engines in round-robin order, and forwards the
resulting mooncake keys back to the controller's sample pool.

NOTE: The inference engine stores tensors in mooncake.
This dispatcher only receives keys and forwards them to training.

Data flow:
  Controller.prompt_buffer → dispatcher._prompt_buffer → Engine.generate()
  → Engine stores tensors in mooncake → returns dict with mooncake_key
  → dispatcher → Controller.sample_pool

Backpressure:
  When the ready pool reaches _max_pool_size, stops admitting new requests.
  Already admitted requests still complete within the bounded engine window;
  transfer storage and ownership are accounted for independently.
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import ray
from ray.exceptions import RayActorError

from aspec.inference.concurrency import effective_engine_concurrency
from aspec.locality import (
    allocate_node_ready_watermarks,
    allocate_node_sample_budgets,
)
from aspec.utils.logging import logger
from aspec.utils.types import InferenceInput, InferenceOutput
from aspec.work_lifecycle import WORK_IDENTITY_KEYS, require_work_identity

MOONCAKE_BACKPRESSURE_POLL_INTERVAL = 0.5  # seconds
MOONCAKE_BACKPRESSURE_LOG_INTERVAL = 5.0  # seconds


def remaining_sample_budget(sample_budget: int | None, scheduled_samples: int) -> int:
    if sample_budget is None:
        return 2**63 - 1
    return max(0, sample_budget - scheduled_samples)


def resolve_training_sample_budget(
    num_train_steps: int | None,
    global_batch_size: int | None,
    *,
    packing_enabled: bool,
    source_work_budget: int | None = None,
) -> int | None:
    """Bound fixed-row inference without assuming samples per packed row."""
    if source_work_budget is not None:
        if source_work_budget <= 0:
            raise ValueError("source_work_budget must be positive")
        return int(source_work_budget)
    if packing_enabled or num_train_steps is None or global_batch_size is None:
        return None
    return int(num_train_steps) * int(global_batch_size)


def should_dispatch_partial_batch(
    *,
    buffered: int,
    batch_size: int,
    pending: int,
    max_concurrent: int,
    remaining_budget: int,
    source_dry: bool,
) -> bool:
    """Dispatch a source tail or the exact final training-budget tail."""
    return (
        0 < buffered < batch_size
        and pending < max_concurrent
        and remaining_budget > 0
        and (source_dry or remaining_budget <= buffered)
    )


def has_ready_pool_capacity(*, pool_size: int, max_pool_size: int) -> bool:
    """A ready watermark stops new work, not completion of bounded in-flight work."""

    if pool_size < 0:
        raise ValueError("ready pool occupancy must be known and non-negative")
    return max_pool_size <= 0 or pool_size < max_pool_size


def effective_prompt_buffer_watermark(
    *, configured: int, batch_size: int, max_concurrent: int
) -> int:
    """Keep enough prompts ready to occupy all runnable engine slots."""

    dispatch_window = batch_size * max_concurrent
    return max(configured, batch_size, dispatch_window)


@dataclass
class EngineLease:
    """One explicitly placed Target-engine concurrency slot."""

    engine: Any
    node: str
    _pool: "EnginePool"
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._release(self.engine, self.node)

    async def __aenter__(self) -> "EngineLease":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


class EnginePool:
    """Fairly lease a bounded number of concurrent slots from each engine."""

    def __init__(
        self,
        engines: list,
        max_concurrent_per_engine: int = 1,
        *,
        engine_nodes: list[str] | None = None,
        planned_nodes: list[str] | None = None,
    ):
        if not engines:
            raise ValueError("engines must be non-empty")
        if max_concurrent_per_engine < 1:
            raise ValueError("max_concurrent_per_engine must be positive")
        if engine_nodes is None:
            engine_nodes = ["local"] * len(engines)
        if len(engine_nodes) != len(engines):
            raise ValueError(
                "engine_nodes must contain one observed node for every engine: "
                f"engines={len(engines)}, engine_nodes={len(engine_nodes)}"
            )
        if any(not isinstance(node, str) or not node for node in engine_nodes):
            raise ValueError("engine_nodes entries must be non-empty strings")

        observed_nodes = set(engine_nodes)
        if planned_nodes is None:
            node_order = list(dict.fromkeys(engine_nodes))
        else:
            node_order = list(dict.fromkeys(planned_nodes))
            if len(node_order) != len(planned_nodes) or any(not node for node in node_order):
                raise ValueError("planned_nodes must contain unique non-empty node identities")
            if observed_nodes != set(node_order):
                raise ValueError(
                    "observed inference engine nodes do not match the Plan: "
                    f"planned={node_order}, observed={sorted(observed_nodes)}"
                )

        self._engines = list(engines)
        self._node_order = node_order
        self._max_concurrent = max_concurrent_per_engine * len(engines)
        self._available_by_node = {node: asyncio.Queue() for node in node_order}
        self._capacity_by_node = {node: 0 for node in node_order}
        self._next_node_index = 0
        for _ in range(max_concurrent_per_engine):
            for engine, node in zip(engines, engine_nodes, strict=True):
                self._available_by_node[node].put_nowait(engine)
                self._capacity_by_node[node] += 1

    def acquire_nowait(self, *, excluded_nodes: set[str] | None = None) -> EngineLease | None:
        """Lease the next available node fairly without relying on actor order."""

        excluded = excluded_nodes or set()
        for offset in range(len(self._node_order)):
            node_index = (self._next_node_index + offset) % len(self._node_order)
            node = self._node_order[node_index]
            available = self._available_by_node[node]
            if node in excluded or available.empty():
                continue
            self._next_node_index = (node_index + 1) % len(self._node_order)
            return EngineLease(available.get_nowait(), node, self)
        return None

    def _release(self, engine: Any, node: str) -> None:
        self._available_by_node[node].put_nowait(engine)

    @asynccontextmanager
    async def lease(self, node: str | None = None):
        """Lease one engine slot while preserving node-level round robin.

        Returning a fast engine to one global FIFO lets that node immediately
        win another request.  With node-local feature affinity, the ready pool
        can then fill with work that cannot form a locality-safe Draft round.
        Choose the node independently from engine completion order, while
        retaining FIFO reuse among the engines placed on that node.
        """

        if node is None:
            while (lease := self.acquire_nowait()) is None:
                await asyncio.sleep(0)
        else:
            if node not in self._available_by_node:
                raise ValueError(f"unknown engine node {node!r}")
            engine = await self._available_by_node[node].get()
            lease = EngineLease(engine, node, self)
        try:
            yield lease.engine
        finally:
            lease.release()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def available(self) -> int:
        return sum(queue.qsize() for queue in self._available_by_node.values())

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._node_order)

    @property
    def available_by_node(self) -> dict[str, int]:
        return {
            node: self._available_by_node[node].qsize() for node in self._node_order
        }

    @property
    def capacity_by_node(self) -> dict[str, int]:
        return dict(self._capacity_by_node)

    def __len__(self) -> int:
        return len(self._engines)


class MetricsCollector:
    """Collects per-sample metrics from engine outputs and aggregates on flush."""

    # engine output key → buffer key
    _KEYS = {
        "e2e_latency": "e2e_latencies",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "spec_accept_rate": "spec_accept_rates",
        "spec_accept_length": "spec_accept_lengths",
    }

    # buffer key → (wandb metric prefix, include min/max)
    _AGGREGATIONS = {
        "e2e_latencies": ("inference/e2e_latency", True),
        "prompt_tokens": ("inference/prompt_tokens", False),
        "completion_tokens": ("inference/completion_tokens", False),
        "spec_accept_rates": ("inference/spec_accept_rate", False),
        "spec_accept_lengths": ("inference/spec_accept_length", False),
    }

    def __init__(self):
        self._buffer: dict[str, list] = {k: [] for k in self._AGGREGATIONS}

    def record(self, output: dict) -> None:
        """Record metrics from a single engine output dict."""
        for key, buffer_key in self._KEYS.items():
            if key in output:
                self._buffer[buffer_key].append(output[key])

    def flush(self) -> dict:
        """Return aggregated metrics and reset."""
        metrics = {}
        for buffer_key, (prefix, with_minmax) in self._AGGREGATIONS.items():
            values = self._buffer[buffer_key]
            if not values:
                continue
            metrics[f"{prefix}_avg"] = sum(values) / len(values)
            if with_minmax:
                metrics[f"{prefix}_min"] = min(values)
                metrics[f"{prefix}_max"] = max(values)
        self._buffer = {k: [] for k in self._AGGREGATIONS}
        return metrics


def producer_batch_rate_metrics(
    batch_times: list[tuple[int, float, float]],
    *,
    perf_window_seconds: float,
    max_concurrent: int,
    now: float | None = None,
) -> dict:
    """Return a backpressure-coupled producer rate estimate.

    The public name intentionally avoids ``capacity``: these samples are taken
    from the live end-to-end pipeline rather than an isolated service profile.
    """

    cutoff = (time.time() if now is None else now) - perf_window_seconds
    recent = [
        (samples, elapsed)
        for samples, elapsed, finished_at in batch_times
        if finished_at > cutoff
    ]
    if not recent:
        return {}

    total_samples = sum(samples for samples, _ in recent)
    total_time = sum(elapsed for _, elapsed in recent)
    if total_time <= 0.001:
        return {}

    per_slot_rate = total_samples / total_time
    return {
        "perf/producer_batch_rate_estimate": per_slot_rate * max_concurrent,
        "perf/infer_batch_time": total_time / len(recent),
    }


@ray.remote
class AsyncInferenceManager:
    """Dispatches inference batches to distributed Ray engine actors.

    Main loop::

        Controller ──get_prompts──→ [_prompt_buffer]
                                         │
                        ┌────────────────┘
                        ▼
                  _dispatch_ready_batches ──→ EnginePool ──→ engine.generate()
                        │                        ▲
                        │ (backpressure:          │ semaphore
                        │  pool full → wait)      │
                        ▼                         │
                  _collect_completed ◄─── _pending_tasks
                        │
                        ▼
                  _forward_results ──→ Controller.push_inference_results()

    Public API (called via Ray remote):
      run(), stop(), flush_metrics(), get_status(), get_pool_status()
    """

    def __init__(
        self,
        args,
        controller,
        inference_engines: list | None = None,
        inference_engine_nodes: list[str] | None = None,
        max_concurrent_batches: int = 1,
    ):
        self.args = args
        self.controller = controller

        self._metrics = MetricsCollector()

        self._prompt_buffer: deque[InferenceInput] = deque()
        self._batch_size = getattr(args, "inference_batch_size", 8)
        self._max_pool_size = getattr(args, "max_sample_pool_size", 0)
        source_slots = None
        if getattr(args, "mooncake_data_plane", "store") == "transfer_engine":
            source_slots = getattr(args, "mooncake_source_gpu_ring_slots", None)
            if source_slots is None:
                source_slots = getattr(args, "mooncake_gpu_ring_slots", 2)
        slots_per_sample = int(getattr(args, "draft_context_parallel_size", 1))
        per_engine_concurrency = effective_engine_concurrency(
            configured=max_concurrent_batches,
            batch_size=self._batch_size,
            source_slots=source_slots,
            slots_per_sample=slots_per_sample,
        )
        planned_nodes: list[str] | None = None
        node_dispatch_demand: dict[str, int] = {}
        planned_node_watermarks: dict[str, int] = {}
        if getattr(args, "mooncake_feature_affinity", "any") == "node_local_required":
            locality_domains = list(
                getattr(args, "aspec_locality_domains", None) or ()
            )
            if not locality_domains:
                raise ValueError(
                    "node_local_required inference dispatch needs compiled locality domains"
                )
            planned_nodes = []
            for domain in locality_domains:
                node = str(domain.get("node_address", ""))
                demand = int(domain.get("dispatch_samples_per_microstep", 0))
                target_ranks = list(domain.get("target_engine_ranks") or ())
                if not node or demand < 1 or not target_ranks:
                    raise ValueError(
                        "node_local_required locality domains need a node address, "
                        "positive Draft demand, and Target engines"
                    )
                planned_nodes.append(node)
                node_dispatch_demand[node] = demand
                planned_node_watermarks[node] = int(
                    domain.get("ready_watermark", 0)
                )
            if inference_engine_nodes is None:
                raise ValueError(
                    "node_local_required inference dispatch needs observed engine-node placement"
                )
        self._engines = EnginePool(
            inference_engines or [],
            per_engine_concurrency,
            engine_nodes=inference_engine_nodes,
            planned_nodes=planned_nodes,
        )
        self._node_ready_watermarks = (
            allocate_node_ready_watermarks(
                self._max_pool_size,
                node_dispatch_demand,
            )
            if planned_nodes is not None
            else {}
        )
        if planned_node_watermarks and (
            self._node_ready_watermarks != planned_node_watermarks
        ):
            raise ValueError(
                "runtime node-local ready credits do not match the compiled Plan: "
                f"planned={planned_node_watermarks}, "
                f"runtime={self._node_ready_watermarks}"
            )
        configured_watermark = getattr(args, "inference_buffer_threshold", 32)
        self._buffer_low_watermark = effective_prompt_buffer_watermark(
            configured=configured_watermark,
            batch_size=self._batch_size,
            max_concurrent=self._engines.max_concurrent,
        )
        self._refill_size = getattr(args, "inference_fetch_batch", 1)

        self._running = True
        self._pending_tasks: set[asyncio.Task] = set()
        self._task_sizes: dict[asyncio.Task, int] = {}
        self._task_nodes: dict[asyncio.Task, str] = {}
        self._inflight_samples = 0
        self._inflight_samples_by_node = {
            node: 0 for node in self._node_ready_watermarks
        }
        self._peak_inflight_samples = 0
        self._peak_inflight_samples_by_node = {
            node: 0 for node in self._node_ready_watermarks
        }
        self._peak_admitted_ready_samples = 0
        self._peak_admitted_ready_by_node = {
            node: 0 for node in self._node_ready_watermarks
        }
        self._peak_reserved_samples = 0
        self._peak_reserved_by_node = {
            node: 0 for node in self._node_ready_watermarks
        }
        self._capacity_rejections = 0
        self._capacity_rejections_by_node = {
            node: 0 for node in self._node_ready_watermarks
        }
        self._source_dry = False

        self._defer_tokenization = getattr(args, "defer_tokenization", False)
        self._return_hidden_states = getattr(args, "compute_logits_in_trainer", True)
        self._train_with_decode = getattr(args, "train_with_decode", False)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)
        self._perf_window_seconds: float = 10.0
        # Each entry: (num_samples, elapsed_seconds, finish_timestamp)
        self._batch_times: deque[tuple[int, float, float]] = deque(maxlen=50)

        self._last_pool_full_log_time = 0.0
        self._request_counter = 0
        packing_enabled = bool(getattr(args, "dflash_packing", False))
        self._training_sample_budget = resolve_training_sample_budget(
            getattr(args, "num_train_steps", None),
            getattr(args, "global_batch_size", None),
            packing_enabled=packing_enabled,
            source_work_budget=getattr(args, "source_work_budget", None),
        )
        producer_profile_samples = int(
            getattr(args, "topology_producer_profile_samples", 0)
        )
        self._budget_phase = (
            "producer_profile" if producer_profile_samples > 0 else "training"
        )
        self._sample_budget = (
            producer_profile_samples
            if producer_profile_samples > 0
            else self._training_sample_budget
        )
        self._scheduled_samples = 0
        self._node_dispatch_demand = node_dispatch_demand
        self._sample_budget_by_node = (
            allocate_node_sample_budgets(
                self._sample_budget,
                node_dispatch_demand,
                request_batch_size=self._batch_size,
            )
            if self._node_ready_watermarks
            else {}
        )
        self._scheduled_samples_by_node = {
            node: 0 for node in self._sample_budget_by_node
        }
        self._phase_after_eval: tuple[str, int | None] | None = None

        if self._max_pool_size > 0:
            logger.info(
                "Flow control enabled: ready_watermark=%d, inflight_sample_limit=%d",
                self._max_pool_size,
                self._batch_size * self._engines.max_concurrent,
            )
            if self._node_ready_watermarks:
                logger.info(
                    "Node-local ready credits: %s",
                    self._node_ready_watermarks,
                )
        else:
            logger.warning("Flow control disabled: max_pool_size=0 (unlimited generation)")
        if self._buffer_low_watermark != configured_watermark:
            logger.info(
                "Raised inference prompt buffer low watermark from %d to %d "
                "to feed %d concurrent engine batches",
                configured_watermark,
                self._buffer_low_watermark,
                self._engines.max_concurrent,
            )
        if per_engine_concurrency != max_concurrent_batches:
            logger.info(
                "Limited per-engine inference concurrency from %d to %d by "
                "source ring capacity (%d slots for batch size %d, %d slots per sample)",
                max_concurrent_batches,
                per_engine_concurrency,
                source_slots,
                self._batch_size,
                slots_per_sample,
            )
        if packing_enabled:
            logger.info(
                "Packing enabled: inference follows the source because samples "
                "per optimizer step are dynamic"
            )

    # -- Public API ----------------------------------------------------------

    async def run(self) -> None:
        """Main loop: refill runs in background, dispatch → collect in foreground."""
        logger.info(
            f"AsyncInferenceManager starting ({len(self._engines)} engines, "
            f"max_concurrent={self._engines.max_concurrent})"
        )

        refill_task = asyncio.create_task(self._continuous_refill())

        try:
            while self._running:
                if refill_task.done():
                    if refill_task.cancelled():
                        raise RuntimeError("inference prompt refill task was cancelled")
                    refill_error = refill_task.exception()
                    if refill_error is not None:
                        raise refill_error
                    raise RuntimeError("inference prompt refill task stopped unexpectedly")

                await self._dispatch_ready_batches()

                if not self._pending_tasks:
                    await asyncio.sleep(0.01)
                    continue

                await self._collect_completed()
        finally:
            self._running = False
            if not refill_task.done():
                refill_task.cancel()
                try:
                    await refill_task
                except asyncio.CancelledError:
                    pass
            await self._drain()
            logger.info("AsyncInferenceManager stopped")

    async def _continuous_refill(self) -> None:
        """Background task: continuously refill prompt buffer."""
        while self._running:
            await self._refill_buffer()
            await asyncio.sleep(0.001)  # yield to event loop

    def stop(self) -> None:
        """Signal the main loop to stop after current iteration."""
        self._running = False

    def flush_metrics(self) -> dict:
        """Return aggregated metrics and reset the buffer."""
        metrics = self._metrics.flush()
        if self._enable_perf_metrics:
            metrics.update(self._compute_perf_metrics())
        return metrics

    def _reset_budget(self, phase: str, sample_budget: int | None) -> dict:
        """Reset accounting for a new, non-overlapping producer phase."""

        self._budget_phase = phase
        self._sample_budget = sample_budget
        self._scheduled_samples = 0
        self._sample_budget_by_node = (
            allocate_node_sample_budgets(
                self._sample_budget,
                self._node_dispatch_demand,
                request_batch_size=self._batch_size,
            )
            if self._node_ready_watermarks
            else {}
        )
        self._scheduled_samples_by_node = {
            node: 0 for node in self._sample_budget_by_node
        }
        self._source_dry = False
        self._metrics.flush()
        self._batch_times.clear()
        self._peak_reserved_samples = 0
        self._peak_inflight_samples = 0
        self._peak_admitted_ready_samples = 0
        self._capacity_rejections = 0
        for attribute in (
            "_peak_reserved_by_node",
            "_peak_inflight_samples_by_node",
            "_peak_admitted_ready_by_node",
            "_capacity_rejections_by_node",
        ):
            values = getattr(self, attribute, {})
            for node in values:
                values[node] = 0
        return {
            "phase": self._budget_phase,
            "sample_budget": self._sample_budget,
        }

    def _require_drained_budget(self, label: str) -> None:
        remaining = self._remaining_sample_budget()
        if (
            remaining
            or self._pending_tasks
            or self._inflight_samples
            or self._prompt_buffer
        ):
            raise RuntimeError(
                f"cannot finish {label} before its work drains: "
                f"remaining={remaining}, pending={len(self._pending_tasks)}, "
                f"inflight={self._inflight_samples}, buffered={len(self._prompt_buffer)}"
            )

    def start_eval_budget(self, sample_count: int) -> dict:
        """Give cache generation a budget isolated from producer/training work."""

        if sample_count <= 0:
            raise ValueError("eval sample budget must be positive")
        if self._phase_after_eval is not None:
            raise RuntimeError("eval budget is already active")
        if (
            self._scheduled_samples
            or self._pending_tasks
            or self._inflight_samples
            or self._prompt_buffer
        ):
            raise RuntimeError(
                "cannot start eval budget after producer work has started: "
                f"scheduled={self._scheduled_samples}, "
                f"pending={len(self._pending_tasks)}, "
                f"inflight={self._inflight_samples}, "
                f"buffered={len(self._prompt_buffer)}"
            )
        self._phase_after_eval = (self._budget_phase, self._sample_budget)
        return self._reset_budget("eval", sample_count)

    def finish_eval_budget(self) -> dict:
        """Restore the untouched producer/training budget after eval drains."""

        if self._budget_phase != "eval" or self._phase_after_eval is None:
            raise RuntimeError(
                f"cannot finish eval budget from phase {self._budget_phase!r}"
            )
        self._require_drained_budget("eval")
        phase, sample_budget = self._phase_after_eval
        self._phase_after_eval = None
        return self._reset_budget(phase, sample_budget)

    def start_training_budget(self) -> dict:
        """Start the normal training budget after an isolated producer probe.

        The controller must first consume and release every output generated by
        the producer probe.  Keeping the transition on the inference actor makes
        it impossible to silently mix outstanding producer work into the
        consumer/end-to-end measurement window.
        """

        if self._budget_phase != "producer_profile":
            raise RuntimeError(
                f"cannot start training budget from phase {self._budget_phase!r}"
            )
        remaining = self._remaining_sample_budget()
        if (
            remaining
            or self._pending_tasks
            or self._inflight_samples
            or self._prompt_buffer
        ):
            raise RuntimeError(
                "cannot start training budget before producer work drains: "
                f"remaining={remaining}, pending={len(self._pending_tasks)}, "
                f"inflight={self._inflight_samples}, "
                f"buffered={len(self._prompt_buffer)}"
            )
        return self._reset_budget("training", self._training_sample_budget)

    def _compute_perf_metrics(self) -> dict:
        """Estimate producer batch service rate over a sliding window.

        This metric is sampled while the end-to-end pipeline is running.  It
        can therefore be limited by ready-pool backpressure and must not be
        interpreted as an independently measured Target capacity.  Dedicated
        topology calibration uses ``_run_producer_service_profile`` instead.
        """
        if not self._batch_times:
            return {}

        return producer_batch_rate_metrics(
            list(self._batch_times),
            perf_window_seconds=self._perf_window_seconds,
            max_concurrent=self._engines.max_concurrent,
        )

    async def get_status(self) -> dict:
        """Get current status of inference manager."""
        remaining_budget = self._remaining_sample_budget()
        status = {
            "mode": f"distributed_engines ({len(self._engines)})",
            "budget_phase": self._budget_phase,
            "sample_budget": self._sample_budget,
            "scheduled_samples": self._scheduled_samples,
            "prompt_buffer_size": len(self._prompt_buffer),
            "running": self._running,
            "pending_tasks": len(self._pending_tasks),
            "inflight_samples": self._inflight_samples,
            "inflight_sample_limit": self._batch_size * self._engines.max_concurrent,
            "peak_inflight_samples": self._peak_inflight_samples,
            "peak_admitted_ready_samples": self._peak_admitted_ready_samples,
            "peak_reserved_samples": self._peak_reserved_samples,
            "capacity_rejections": self._capacity_rejections,
            "engine_slots_available": self._engines.available,
            "remaining_sample_budget": remaining_budget,
            "sample_budget_drained": remaining_budget == 0
            and not self._pending_tasks,
            "source_drained": self._source_dry
            and not self._prompt_buffer
            and not self._pending_tasks,
        }
        if self._node_ready_watermarks:
            status.update(
                {
                    "ready_watermark_by_node": dict(self._node_ready_watermarks),
                    "sample_budget_by_node": dict(self._sample_budget_by_node),
                    "scheduled_samples_by_node": dict(
                        self._scheduled_samples_by_node
                    ),
                    "inflight_samples_by_node": dict(self._inflight_samples_by_node),
                    "inflight_sample_limit_by_node": {
                        node: self._batch_size * slots
                        for node, slots in self._engines.capacity_by_node.items()
                    },
                    "peak_inflight_samples_by_node": dict(
                        self._peak_inflight_samples_by_node
                    ),
                    "peak_admitted_ready_by_node": dict(
                        self._peak_admitted_ready_by_node
                    ),
                    "peak_reserved_by_node": dict(self._peak_reserved_by_node),
                    "capacity_rejections_by_node": dict(
                        self._capacity_rejections_by_node
                    ),
                    "engine_slots_available_by_node": self._engines.available_by_node,
                }
            )
        status.update(await self.get_pool_status())
        return status

    async def get_pool_status(self) -> dict:
        """Get scheduler queue and transfer-residency status separately."""
        try:
            pool_size = await self.controller.get_ready_pool_size.remote()
            pool_size_by_node = (
                await self.controller.get_ready_pool_size_by_node.remote()
                if self._node_ready_watermarks
                else {}
            )
            resident_outputs = await self.controller.get_pool_size.remote()
            resident_bytes = await self.controller.get_pool_bytes.remote()
        except Exception:
            pool_size = -1
            pool_size_by_node = {}
            resident_outputs = -1
            resident_bytes = -1

        return {
            "max_pool_size": self._max_pool_size,
            "current_pool_size": pool_size,
            "current_pool_size_by_node": pool_size_by_node,
            "resident_outputs": resident_outputs,
            "resident_bytes": resident_bytes,
            "pool_usage_percent": (
                int(pool_size * 100 / self._max_pool_size)
                if self._max_pool_size > 0 and pool_size >= 0
                else 0
            ),
            "flow_control_enabled": self._max_pool_size > 0,
        }

    # -- Main loop internals -------------------------------------------------

    async def _dispatch_ready_batches(self) -> None:
        """Dispatch batches from buffer, up to the engine concurrency limit."""
        while (
            len(self._prompt_buffer) >= self._batch_size
            and len(self._pending_tasks) < self._engines.max_concurrent
            and self._remaining_sample_budget() > 0
        ):
            dispatch_size = min(self._batch_size, self._remaining_sample_budget())
            if not await self._start_dispatch(dispatch_size):
                break

        remaining = self._remaining_sample_budget()
        if should_dispatch_partial_batch(
            buffered=len(self._prompt_buffer),
            batch_size=self._batch_size,
            pending=len(self._pending_tasks),
            max_concurrent=self._engines.max_concurrent,
            remaining_budget=remaining,
            source_dry=self._source_dry,
        ):
            partial_size = min(self._batch_size, remaining, len(self._prompt_buffer))
            await self._start_dispatch(partial_size)

    async def _start_dispatch(self, dispatch_size: int) -> bool:
        """Atomically acquire one observed engine slot and its pool credit."""

        # An engine releases its execution slot before this manager collects
        # and forwards its output.  That completed task still owns admission
        # credit; do not select its node until the credit has been retired.
        excluded_nodes = {
            node
            for node, inflight in self._inflight_samples_by_node.items()
            if inflight + dispatch_size
            > self._batch_size * self._engines.capacity_by_node[node]
        } if self._node_ready_watermarks else set()
        while True:
            lease = self._engines.acquire_nowait(excluded_nodes=excluded_nodes)
            if lease is None:
                return False
            node = lease.node if self._node_ready_watermarks else None
            if node is not None and self._sample_budget_by_node:
                node_remaining = (
                    self._sample_budget_by_node[node]
                    - self._scheduled_samples_by_node[node]
                )
                if node_remaining < dispatch_size:
                    lease.release()
                    excluded_nodes.add(node)
                    continue
            try:
                admitted = await self._reserve_pool_capacity(
                    dispatch_size, node=node
                )
            except BaseException:
                lease.release()
                raise
            if admitted:
                break
            lease.release()
            excluded_nodes.add(lease.node)

        entries = self._take_batch(dispatch_size)
        self._scheduled_samples += len(entries)
        if node is not None and self._sample_budget_by_node:
            self._scheduled_samples_by_node[node] += len(entries)
        task = asyncio.create_task(self._dispatch_batch(entries, engine_lease=lease))
        self._pending_tasks.add(task)
        self._task_sizes[task] = len(entries)
        if node is not None:
            self._task_nodes[task] = node
        return True

    def _remaining_sample_budget(self) -> int:
        return remaining_sample_budget(self._sample_budget, self._scheduled_samples)

    async def _collect_completed(self) -> None:
        """Wait for at least one dispatch to finish and forward results."""
        # With idle slots, ready/prompt availability can change before the last
        # running request completes. Revisit admission instead of tail-blocking
        # all idle engines; keep the direct completion wait when slots are full.
        timeout = (
            MOONCAKE_BACKPRESSURE_POLL_INTERVAL
            if len(self._pending_tasks) < self._engines.max_concurrent
            else None
        )
        done, self._pending_tasks = await asyncio.wait(
            self._pending_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
        )
        first_error = None
        for task in done:
            reserved = self._task_sizes.pop(task, 0)
            node = self._task_nodes.pop(task, None)
            try:
                await self._forward_results(task.result())
            except BaseException as exc:
                first_error = first_error or exc
            finally:
                self._inflight_samples -= reserved
                if node is not None:
                    self._inflight_samples_by_node[node] -= reserved
        if first_error is not None:
            raise first_error

    async def _drain(self) -> None:
        # Currently drain is only called on shutdown, so we don't need to await pool.
        if self._prompt_buffer:
            logger.info(
                f"Drain: discarding {len(self._prompt_buffer)} buffered prompts on shutdown"
            )
            self._prompt_buffer.clear()
        if self._pending_tasks:
            done, _ = await asyncio.wait(self._pending_tasks, return_when=asyncio.ALL_COMPLETED)
            for task in done:
                reserved = self._task_sizes.pop(task, 0)
                node = self._task_nodes.pop(task, None)
                try:
                    await self._forward_results(task.result())
                except BaseException as exc:
                    logger.warning(f"Pending engine dispatch failed during drain: {exc}")
                finally:
                    self._inflight_samples -= reserved
                    if node is not None:
                        self._inflight_samples_by_node[node] -= reserved
            self._pending_tasks.clear()

    # -- Buffer management ---------------------------------------------------

    async def _refill_buffer(self) -> None:
        """Fetch prompts from controller if buffer is below watermark.

        Requests enough prompts to fill the buffer up to the watermark in one
        call, avoiding multiple round-trips to the controller.
        """
        remaining = self._remaining_sample_budget()
        if remaining <= len(self._prompt_buffer):
            return
        if len(self._prompt_buffer) >= self._buffer_low_watermark:
            return

        if self._max_pool_size > 0:
            if self._node_ready_watermarks:
                pool_sizes = await self.controller.get_ready_pool_size_by_node.remote()
                pool_full = all(
                    int(pool_sizes.get(node, 0)) >= watermark
                    for node, watermark in self._node_ready_watermarks.items()
                )
                pool_description = f"by_node={pool_sizes}/{self._node_ready_watermarks}"
            else:
                pool_size = await self.controller.get_ready_pool_size.remote()
                pool_full = pool_size >= self._max_pool_size
                pool_description = f"{pool_size}/{self._max_pool_size}"
            if pool_full:
                now = time.monotonic()
                if now - self._last_pool_full_log_time >= 2.0:
                    self._last_pool_full_log_time = now
                    logger.debug(
                        f"Skipping prompt fetch: pool full ({pool_description})"
                    )
                return

        need = self._buffer_low_watermark - len(self._prompt_buffer)
        fetch_size = min(
            max(need, self._refill_size),
            remaining - len(self._prompt_buffer),
        )
        entries = await self.controller.get_prompts.remote(fetch_size)
        if entries:
            self._prompt_buffer.extend(entries)
            self._source_dry = False
            logger.debug(f"Fetched {len(entries)} prompts, buffer_size={len(self._prompt_buffer)}")
        else:
            self._source_dry = True

    def _take_batch(self, n: int) -> list[InferenceInput]:
        """Pop up to n entries from internal buffer."""
        n = min(n, len(self._prompt_buffer))
        return [self._prompt_buffer.popleft() for _ in range(n)]

    # -- Backpressure --------------------------------------------------------

    async def _reserve_pool_capacity(
        self,
        incoming_samples: int,
        *,
        node: str | None = None,
    ) -> bool:
        """Gate on ready outputs while accounting for independently bounded requests.

        Reuse TorchSpec's ready-watermark policy without blocking collection of
        completed requests. Pending outputs may cross the watermark, but cannot
        exceed one engine dispatch window above the last admitted ready size.
        Transfer storage and leases remain independently bounded by the data plane.
        """

        if incoming_samples <= 0:
            return False
        inflight_limit = self._batch_size * self._engines.max_concurrent
        if incoming_samples > self._batch_size or (
            self._inflight_samples + incoming_samples > inflight_limit
        ):
            raise RuntimeError("inference admission exceeded the engine dispatch window")
        pool_size = 0
        if self._max_pool_size > 0:
            if self._node_ready_watermarks:
                if node not in self._node_ready_watermarks:
                    raise RuntimeError(
                        f"engine lease has no planned locality credit: node={node!r}"
                    )
                pool_sizes = await self.controller.get_ready_pool_size_by_node.remote()
                unknown_nodes = set(pool_sizes) - set(self._node_ready_watermarks)
                if unknown_nodes:
                    raise RuntimeError(
                        "ready pool contains unplanned source nodes: "
                        f"{sorted(unknown_nodes)}"
                    )
                node_pool_size = int(pool_sizes.get(node, 0))
                node_watermark = self._node_ready_watermarks[node]
                if not has_ready_pool_capacity(
                    pool_size=node_pool_size,
                    max_pool_size=node_watermark,
                ):
                    self._capacity_rejections += 1
                    self._capacity_rejections_by_node[node] += 1
                    return False
                pool_size = sum(int(value) for value in pool_sizes.values())
                self._peak_admitted_ready_by_node[node] = max(
                    self._peak_admitted_ready_by_node[node], node_pool_size
                )
            else:
                pool_size = await self.controller.get_ready_pool_size.remote()
                if not has_ready_pool_capacity(
                    pool_size=pool_size, max_pool_size=self._max_pool_size
                ):
                    self._capacity_rejections += 1
                    return False
            self._peak_admitted_ready_samples = max(
                self._peak_admitted_ready_samples, pool_size
            )
        if node is not None:
            node_limit = self._batch_size * self._engines.capacity_by_node[node]
            if self._inflight_samples_by_node[node] + incoming_samples > node_limit:
                raise RuntimeError(
                    "node-local inference admission exceeded the engine dispatch window: "
                    f"node={node!r}, "
                    f"inflight={self._inflight_samples_by_node[node]}, "
                    f"incoming={incoming_samples}, limit={node_limit}"
                )
        self._inflight_samples += incoming_samples
        self._peak_inflight_samples = max(
            self._peak_inflight_samples, self._inflight_samples
        )
        self._peak_reserved_samples = max(
            self._peak_reserved_samples,
            pool_size + self._inflight_samples,
        )
        if node is not None:
            self._inflight_samples_by_node[node] += incoming_samples
            self._peak_inflight_samples_by_node[node] = max(
                self._peak_inflight_samples_by_node[node],
                self._inflight_samples_by_node[node],
            )
            node_ready = (
                int(pool_sizes.get(node, 0)) if self._max_pool_size > 0 else 0
            )
            self._peak_reserved_by_node[node] = max(
                self._peak_reserved_by_node[node],
                node_ready + self._inflight_samples_by_node[node],
            )
        return True

    async def _await_pool_capacity(self) -> None:
        """Block until sample pool has capacity."""
        if self._max_pool_size <= 0:
            return

        pool_size = await self.controller.get_ready_pool_size.remote()
        if pool_size < self._max_pool_size:
            return

        wait_start = time.time()
        last_log_time = wait_start

        logger.warning(
            f"Sample pool full, pausing generation: pool_size={pool_size}/{self._max_pool_size}"
        )

        while pool_size >= self._max_pool_size and self._running:
            await asyncio.sleep(MOONCAKE_BACKPRESSURE_POLL_INTERVAL)
            pool_size = await self.controller.get_ready_pool_size.remote()

            now = time.time()
            if now - last_log_time >= MOONCAKE_BACKPRESSURE_LOG_INTERVAL:
                wait_duration = now - wait_start
                logger.info(
                    f"Still waiting for pool capacity (waited {wait_duration:.1f}s): "
                    f"pool_size={pool_size}/{self._max_pool_size}"
                )
                last_log_time = now

        wait_duration = time.time() - wait_start
        if not self._running:
            logger.info(f"Pool wait aborted (shutdown requested) after {wait_duration:.1f}s")
            return
        logger.info(
            f"Pool capacity available after {wait_duration:.1f}s, "
            f"pool_size={pool_size}/{self._max_pool_size}"
        )

    # -- Engine dispatch -----------------------------------------------------

    def _prepare_engine_inputs(self, entries: list[InferenceInput]) -> dict:
        """Pack entries into engine.generate() kwargs."""
        packed_loss_mask_list = [e.packed_loss_mask for e in entries]
        data_ids = [e.data_id for e in entries]
        multimodal_inputs = [e.multimodal_inputs for e in entries]
        has_multimodal = any(m is not None for m in multimodal_inputs)

        if self._defer_tokenization:
            input_ids_ref = None
            packed_loss_mask_list = None
            if not all(e.formatted_prompt is not None for e in entries):
                raise ValueError(
                    "formatted_prompt is required when defer_tokenization is True"
                )
            formatted_prompts = [e.formatted_prompt for e in entries]
        else:
            input_ids_ref = ray.put([e.input_ids for e in entries])
            formatted_prompts = None

        return {
            "data_ids": data_ids,
            "input_ids_ref": input_ids_ref,
            "packed_loss_mask_list": packed_loss_mask_list,
            "formatted_prompts": formatted_prompts,
            "return_last_hidden_states": self._return_hidden_states,
            "return_logits": not self._return_hidden_states,
            "multimodal_inputs": multimodal_inputs if has_multimodal else None,
        }

    async def _dispatch_batch(
        self,
        entries: list[InferenceInput],
        *,
        engine_lease: EngineLease | None = None,
    ) -> list[tuple[InferenceInput, dict | Exception]]:
        """Send a batch to the next engine and return (entry, output) pairs."""
        if engine_lease is None:
            engine_context = self._engines.lease()
        else:
            @asynccontextmanager
            async def leased_engine():
                async with engine_lease as lease:
                    yield lease.engine

            engine_context = leased_engine()

        async with engine_context as engine:
            kwargs = self._prepare_engine_inputs(entries)
            request_prefix = f"request_{self._request_counter}"
            self._request_counter += 1
            request_ids = [
                f"{request_prefix}_{index}" for index in range(len(entries))
            ]
            for entry, request_id in zip(entries, request_ids, strict=True):
                entry.metadata = dict(entry.metadata or {})
                entry.metadata["request_id"] = request_id

            try:
                if self._enable_perf_metrics:
                    t0 = time.time()
                generate_method = (
                    engine.generate_with_decode if self._train_with_decode else engine.generate
                )
                generate_kwargs = dict(
                    data_id=kwargs["data_ids"],
                    input_ids_ref=kwargs["input_ids_ref"],
                    packed_loss_mask_list=kwargs["packed_loss_mask_list"],
                    formatted_prompts=kwargs["formatted_prompts"],
                    return_last_hidden_states=kwargs["return_last_hidden_states"],
                    return_logits=kwargs["return_logits"],
                    multimodal_inputs=kwargs["multimodal_inputs"],
                )
                if (
                    getattr(self.args, "inference_engine_type", "sgl") == "sgl"
                    and not self._train_with_decode
                ):
                    generate_kwargs["request_ids"] = request_ids
                outputs = await generate_method.remote(**generate_kwargs)
                if self._enable_perf_metrics:
                    now = time.time()
                    self._batch_times.append((len(entries), now - t0, now))
                if len(outputs) != len(entries):
                    logger.error(
                        f"Engine returned {len(outputs)} results for "
                        f"{len(entries)} entries (expected equal)"
                    )
                    err = ValueError(f"output count mismatch: {len(outputs)} vs {len(entries)}")
                    return [(entry, err) for entry in entries]
                if engine_lease is not None and self._node_ready_watermarks:
                    invalid_sources = [
                        (output.get("metadata") or {}).get("source_node_ip")
                        for output in outputs
                        if (output.get("metadata") or {}).get("source_node_ip")
                        != engine_lease.node
                    ]
                    if invalid_sources:
                        raise RuntimeError(
                            "Target engine output node does not match its observed placement: "
                            f"leased_node={engine_lease.node!r}, "
                            f"output_nodes={invalid_sources!r}"
                        )
                return list(zip(entries, outputs, strict=True))
            except RayActorError as e:
                logger.critical(f"Engine actor died, terminating inference manager: {e}")
                self._running = False
                await self.controller.set_inference_error.remote(str(e))
                raise
            except Exception as e:
                import traceback

                logger.error(f"Engine dispatch failed: {e}\n{traceback.format_exc()}")
                return [(entry, e) for entry in entries]

    # -- Result processing ---------------------------------------------------

    def _parse_engine_output(self, entry: InferenceInput, output: dict) -> InferenceOutput | None:
        """Convert engine output dict to InferenceOutput."""
        if not isinstance(output, dict) or "mooncake_key" not in output:
            logger.debug(
                f"Skipping invalid engine output for data_id={entry.data_id}: {type(output)}"
            )
            return None

        metadata = dict(entry.metadata or {})
        require_work_identity(metadata)
        backend_metadata = dict(output.get("metadata", {}) or {})
        for key in WORK_IDENTITY_KEYS:
            if key in backend_metadata and backend_metadata[key] != metadata[key]:
                raise ValueError(
                    f"Engine output attempted to change scheduler-owned {key} "
                    f"for data_id={entry.data_id}"
                )
            backend_metadata.pop(key, None)
        metadata.update(backend_metadata)
        self._metrics.record(output)

        return InferenceOutput(
            data_id=entry.data_id,
            mooncake_key=output["mooncake_key"],
            tensor_shapes=output.get("tensor_shapes", {}),
            tensor_dtypes=output.get("tensor_dtypes", {}),
            packed_loss_mask=output.get("packed_loss_mask", entry.packed_loss_mask),
            metadata=metadata,
        )

    async def _forward_results(self, results: list[tuple[InferenceInput, Any | Exception]]) -> int:
        """Parse results and forward to controller. Returns success count."""
        inference_results = []
        failures = []

        for entry, result in results:
            if isinstance(result, Exception):
                failures.append(f"{entry.data_id}: {type(result).__name__}: {result}")
                continue

            try:
                inference_result = self._parse_engine_output(entry, result)
            except Exception as exc:
                failures.append(
                    f"{entry.data_id}: invalid engine output: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if inference_result is not None:
                inference_results.append(inference_result)
            else:
                failures.append(f"{entry.data_id}: invalid engine output")

        if inference_results:
            await self.controller.push_inference_results.remote(inference_results)
            logger.debug(f"Forwarded {len(inference_results)} results to controller")

        if failures:
            message = "Inference batch lost required work: " + "; ".join(failures[:8])
            self._running = False
            await self.controller.set_inference_error.remote(message)
            raise RuntimeError(message)

        return len(inference_results)
