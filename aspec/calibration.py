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

"""Bounded candidate compilation and evidence-gated calibration selection.

The selection rules are adapted from AngelSpec's planner. Candidate compilation
preserves the base plan's admitted attention, context-parallel, Target-output,
and transport capabilities while varying only expert-approved capacity and
role-split dimensions. This module prepares and evaluates plans; it never
starts GPU work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aspec.config.draft_execution import is_direct_cp_backend
from aspec.inference.concurrency import direct_output_sample_capacity
from aspec.planning import ExecutionPlan

CALIBRATION_PLAN_SCHEMA_VERSION = "aspec.calibration-plan/v0alpha1"
CALIBRATION_REPORT_SCHEMA_VERSION = "aspec.calibration-report/v0alpha1"


def _identity(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def candidate_identity_from_plan(plan: Mapping[str, Any]) -> str:
    """Identify a candidate by complete execution semantics, not its run directory.

    ``ExecutionPlan.plan_id`` intentionally includes artifact locations, so two
    repeats written to different output directories have different Plan ids.
    Calibration needs the opposite grouping rule: all workload and execution
    decisions must match, while the destination used to persist one attempt is
    allowed to differ.  Hashing the canonical Plan payload after removing only
    artifact paths gives that identity and avoids the hand-maintained partial
    candidate id used by the earlier calibration interface.
    """

    policy = copy.deepcopy(dict(plan["policy"]))
    artifacts = copy.deepcopy(dict(policy.get("artifacts") or {}))
    artifacts.pop("output_dir", None)
    policy["artifacts"] = artifacts
    payload = {
        "workload": copy.deepcopy(plan["workload"]),
        "policy": policy,
        "capabilities": copy.deepcopy(plan.get("capabilities", [])),
        "fallbacks": copy.deepcopy(plan.get("fallbacks", [])),
        "omissions": copy.deepcopy(plan.get("omissions", [])),
    }
    return _identity(payload)


@dataclass(frozen=True)
class CalibrationSearchSpace:
    """Expert-approved dimensions for one bounded calibration exercise."""

    trainer_gpu_counts: tuple[int, ...]
    micro_batch_sizes: tuple[int, ...] = (1,)
    inference_batch_sizes: tuple[int, ...] = (1,)
    max_concurrent_batches: tuple[int, ...] = (1,)
    pool_headroom_steps: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        dimensions = asdict(self)
        empty = [name for name, values in dimensions.items() if not values]
        if empty:
            raise ValueError(f"calibration search dimensions cannot be empty: {empty}")
        invalid = {
            name: values
            for name, values in dimensions.items()
            if any(not isinstance(value, int) or value < 0 for value in values)
        }
        if invalid:
            raise ValueError(
                f"calibration search dimensions must be non-negative integers: {invalid}"
            )
        positive = {
            "trainer_gpu_counts",
            "micro_batch_sizes",
            "inference_batch_sizes",
            "max_concurrent_batches",
        }
        invalid_positive = {
            name: dimensions[name]
            for name in positive
            if any(value < 1 for value in dimensions[name])
        }
        if invalid_positive:
            raise ValueError(
                f"calibration execution dimensions must be positive: {invalid_positive}"
            )


@dataclass(frozen=True)
class CalibrationCandidate:
    attention_backend: str
    context_parallel_size: int
    data_plane: str
    target_prefill_memory_optimized: bool
    total_gpus: int
    trainer_gpus: int
    target_gpus: int
    target_gpus_per_engine: int
    target_engines: int
    micro_batch_size: int
    accumulation_steps: int
    global_batch_size: int
    inference_batch_size: int
    max_concurrent_batches: int
    sample_pool_size: int
    packing: bool

    @property
    def candidate_id(self) -> str:
        target_output = (
            "direct" if self.target_prefill_memory_optimized else "buffered"
        )
        return (
            f"{self.attention_backend}-cp{self.context_parallel_size}"
            f"-{self.data_plane}-{target_output}"
            f"-t{self.target_gpus}-tp{self.target_gpus_per_engine}"
            f"-r{self.trainer_gpus}-m{self.micro_batch_size}"
            f"-a{self.accumulation_steps}-b{self.inference_batch_size}"
            f"-c{self.max_concurrent_batches}-p{self.sample_pool_size}"
        )

    def overrides(self) -> list[str]:
        return [
            f"training.attention_backend={self.attention_backend}",
            f"training.draft_context_parallel_size={self.context_parallel_size}",
            f"mooncake.data_plane={self.data_plane}",
            "inference.target_prefill_memory_optimized="
            f"{str(self.target_prefill_memory_optimized).lower()}",
            f"training.training_num_gpus_per_node={self.trainer_gpus}",
            f"training.micro_batch_size={self.micro_batch_size}",
            f"training.draft_accumulation_steps={self.accumulation_steps}",
            f"training.max_concurrent_batches={self.max_concurrent_batches}",
            f"inference.inference_num_gpus={self.target_gpus}",
            f"inference.inference_num_gpus_per_node={self.target_gpus}",
            f"inference.inference_batch_size={self.inference_batch_size}",
            f"inference.inference_fetch_batch={self.sample_pool_size}",
            f"inference.inference_buffer_threshold={self.sample_pool_size}",
            f"inference.max_sample_pool_size={self.sample_pool_size}",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            **asdict(self),
            "overrides": self.overrides(),
        }


@dataclass(frozen=True)
class CalibrationPlan:
    base_plan_id: str
    workload_id: str
    total_gpus: int
    search_space: CalibrationSearchSpace
    candidates: tuple[CalibrationCandidate, ...]
    rejection_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_PLAN_SCHEMA_VERSION,
            "base_plan_id": self.base_plan_id,
            "workload_id": self.workload_id,
            "total_gpus": self.total_gpus,
            "search_space": asdict(self.search_space),
            "candidate_count": len(self.candidates),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def compile_calibration_plan(
    base_plan: ExecutionPlan,
    *,
    total_gpus: int,
    search_space: CalibrationSearchSpace,
) -> CalibrationPlan:
    """Compile only candidates supported by the current executable path."""
    if total_gpus < 2:
        raise ValueError("calibration requires at least two GPUs")
    policy = base_plan.policy
    if policy.parallelism["training"]["sequence_parallel"] != 1:
        raise ValueError("bounded calibration currently requires sequence_parallel=1")
    attention_backend = policy.parallelism["training"]["attention_backend"]
    context_parallel_size = int(
        policy.parallelism["training"].get("context_parallel", 1)
    )
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be positive")
    data_plane = policy.capacity["transfer_storage"]["data_plane"]
    target_direct = policy.pipeline.get("target_feature_output") == "direct_to_transport"
    target_gpus_per_engine = policy.resources["inference"]["gpus_per_engine"]
    global_batch_size = policy.pipeline["global_batch_size"]
    packing = policy.pipeline["packed_rows"]
    packing_min_samples = (
        policy.pipeline.get("packing_min_samples_per_dispatch", 0) if packing else 0
    )
    prefetch_depth = policy.pipeline["prefetch_depth"]
    transfer_capacity = policy.capacity["transfer_storage"]
    common_slots = int(transfer_capacity.get("gpu_ring_slots") or 2)
    source_slots = transfer_capacity.get("source_gpu_ring_slots")
    destination_slots = transfer_capacity.get("destination_gpu_ring_slots")
    source_slots = common_slots if source_slots is None else int(source_slots)
    destination_slots = (
        common_slots if destination_slots is None else int(destination_slots)
    )
    rejected: Counter[str] = Counter()
    candidates: dict[str, CalibrationCandidate] = {}

    for trainer_gpus in search_space.trainer_gpu_counts:
        target_gpus = total_gpus - trainer_gpus
        if trainer_gpus <= 0 or target_gpus <= 0:
            rejected["role_split_out_of_range"] += 1
            continue
        if trainer_gpus % context_parallel_size:
            rejected["trainer_gpus_not_divisible_by_context_parallel"] += 1
            continue
        if target_gpus % target_gpus_per_engine:
            rejected["target_gpus_not_divisible_by_engine_width"] += 1
            continue
        target_engines = target_gpus // target_gpus_per_engine
        logical_data_parallel = trainer_gpus // context_parallel_size
        for micro_batch_size in search_space.micro_batch_sizes:
            if packing and micro_batch_size != 1:
                rejected["packing_requires_micro_batch_one"] += 1
                continue
            dispatch_width = logical_data_parallel * micro_batch_size
            if global_batch_size % dispatch_width:
                rejected["global_batch_not_divisible_by_dispatch"] += 1
                continue
            slots_per_sample = (
                context_parallel_size if is_direct_cp_backend(attention_backend) else 1
            )
            if target_direct and direct_output_sample_capacity(
                engine_count=target_engines,
                source_slots=source_slots,
                slots_per_sample=slots_per_sample,
            ) < dispatch_width:
                rejected["source_ring_below_direct_dispatch"] += 1
                continue
            if (
                data_plane == "transfer_engine"
                and context_parallel_size > 1
                and destination_slots < micro_batch_size
            ):
                rejected["destination_ring_below_micro_batch"] += 1
                continue
            accumulation_steps = global_batch_size // dispatch_width
            queued_dispatches = max(accumulation_steps, prefetch_depth)
            dispatch_source_minimum = max(dispatch_width, packing_min_samples)
            for inference_batch_size in search_space.inference_batch_sizes:
                slots_per_request = inference_batch_size * slots_per_sample
                if (
                    data_plane == "transfer_engine"
                    and source_slots < slots_per_request
                ):
                    rejected["source_ring_below_request_batch"] += 1
                    continue
                for concurrency in search_space.max_concurrent_batches:
                    if (
                        data_plane == "transfer_engine"
                        and concurrency > source_slots // slots_per_request
                    ):
                        rejected["source_ring_limits_concurrency"] += 1
                        continue
                    structural_pool = max(
                        dispatch_source_minimum * queued_dispatches,
                        inference_batch_size,
                        target_engines * inference_batch_size * concurrency,
                    )
                    for headroom in search_space.pool_headroom_steps:
                        pool_size = structural_pool + headroom * dispatch_width
                        candidate = CalibrationCandidate(
                            attention_backend=attention_backend,
                            context_parallel_size=context_parallel_size,
                            data_plane=data_plane,
                            target_prefill_memory_optimized=target_direct,
                            total_gpus=total_gpus,
                            trainer_gpus=trainer_gpus,
                            target_gpus=target_gpus,
                            target_gpus_per_engine=target_gpus_per_engine,
                            target_engines=target_engines,
                            micro_batch_size=micro_batch_size,
                            accumulation_steps=accumulation_steps,
                            global_batch_size=global_batch_size,
                            inference_batch_size=inference_batch_size,
                            max_concurrent_batches=concurrency,
                            sample_pool_size=pool_size,
                            packing=packing,
                        )
                        candidates[candidate.candidate_id] = candidate

    ordered = tuple(
        sorted(
            candidates.values(),
            key=lambda row: (
                row.trainer_gpus,
                row.micro_batch_size,
                row.inference_batch_size,
                row.max_concurrent_batches,
                row.sample_pool_size,
            ),
        )
    )
    return CalibrationPlan(
        base_plan_id=base_plan.plan_id,
        workload_id=_identity(asdict(base_plan.workload)),
        total_gpus=total_gpus,
        search_space=search_space,
        candidates=ordered,
        rejection_counts=dict(rejected),
    )


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    evidence_path: str
    plan_id: str | None
    workload_id: str | None
    environment_id: str
    hardware_cohort_id: str
    committed_work_per_second: float
    optimizer_steps: int
    peak_resident_bytes: int
    planned_sample_pool_size: int
    completed: bool
    finite_training_signals: bool
    lifecycle_healthy: bool
    required_capabilities_active: bool

    @property
    def valid(self) -> bool:
        return (
            self.completed
            and self.finite_training_signals
            and self.lifecycle_healthy
            and self.required_capabilities_active
            and math.isfinite(self.committed_work_per_second)
            and self.committed_work_per_second > 0
            and self.optimizer_steps > 0
            and self.peak_resident_bytes >= 0
            and self.workload_id is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


def _candidate_id_from_plan(plan: Mapping[str, Any]) -> str:
    return candidate_identity_from_plan(plan)


def _hardware_cohort_identity(evidence: Mapping[str, Any]) -> str:
    runtime = evidence.get("observations", {}).get("runtime", {})
    accelerators = runtime.get("accelerators") or {}
    gpus = accelerators.get("gpus") or []
    if gpus:
        gpu_model = accelerators.get("hardware_label")
        hardware = {
            "visible_gpu_count": accelerators.get("visible_gpu_count"),
            "gpus": sorted(
                (
                    {
                        "gpu_model": gpu_model or row.get("reported_name"),
                        "memory_mib": row.get("memory_mib"),
                        "compute_capability": row.get("compute_capability"),
                    }
                    for row in gpus
                ),
                key=lambda row: json.dumps(row, sort_keys=True),
            ),
        }
    else:
        hardware = {
            "ray_accelerators": {
                key: value
                for key, value in runtime.get("ray", {})
                .get("cluster_resources", {})
                .items()
                if key == "GPU" or key.startswith("accelerator_type:")
            }
        }
    return _identity(hardware)


def _environment_identity(evidence: Mapping[str, Any]) -> str:
    source = evidence.get("source", {})
    runtime = evidence.get("observations", {}).get("runtime", {})
    host = source.get("host") or {}
    accelerators = runtime.get("accelerators") or {}
    backends = {
        name: {
            key: value
            for key, value in (identity or {}).items()
            if key != "path"
        }
        for name, identity in (source.get("backends") or {}).items()
    }
    return _identity(
        {
            "aspec": source.get("aspec"),
            "backends": backends,
            "packages": source.get("packages"),
            "platform": {
                key: host.get(key)
                for key in ("system", "release", "machine", "python")
            },
            "driver_versions": sorted(
                {
                    row.get("driver_version")
                    for row in accelerators.get("gpus") or []
                    if row.get("driver_version") is not None
                }
            ),
            "hardware_cohort_id": _hardware_cohort_identity(evidence),
        }
    )


def load_candidate_measurement(
    evidence_path: str | Path,
) -> CandidateMeasurement:
    path = Path(evidence_path).expanduser().resolve()
    evidence = json.loads(path.read_text(encoding="utf-8"))
    plan_path = path.with_name("plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    if plan is not None:
        declared_plan_id = plan.get("plan_id")
        plan_payload = {key: value for key, value in plan.items() if key != "plan_id"}
        if declared_plan_id != _identity(plan_payload):
            raise ValueError(
                f"adjacent plan content does not match its declared identity: {plan_path}"
            )
        if declared_plan_id != evidence.get("plan_id"):
            raise ValueError(
                f"evidence and adjacent plan identities do not match: {path}"
            )
    training = evidence.get("observations", {}).get("training") or {}
    cleanup = evidence.get("observations", {}).get("cleanup") or {}
    performance = training.get("performance") or {}
    lifecycle = cleanup.get("work_lifecycle") or {}
    final_metrics = training.get("final_metrics") or {}
    loss = final_metrics.get("train/avg_loss")
    grad_norm = final_metrics.get("train/grad_norm")
    finite_training = (
        isinstance(loss, (int, float))
        and isinstance(grad_norm, (int, float))
        and math.isfinite(loss)
        and math.isfinite(grad_norm)
    )
    capabilities = evidence.get("capabilities", [])
    capability_states = {
        item.get("name"): item.get("effective_state") for item in capabilities
    }
    required_names = {
        item.get("name")
        for item in (plan or {}).get("capabilities", [])
        if item.get("requirement") == "required"
    }
    required_active = all(
        capability_states.get(name) == "active" for name in required_names
    )
    performance_active = any(
        item.get("name") == "performance_measurement"
        and item.get("effective_state") == "active"
        for item in capabilities
    )
    candidate_id = _candidate_id_from_plan(plan) if plan is not None else "missing-plan"
    sample_pool_size = (
        (plan or {})
        .get("policy", {})
        .get("capacity", {})
        .get("host_orchestrator", {})
        .get("max_sample_pool_size", -1)
    )
    return CandidateMeasurement(
        candidate_id=candidate_id,
        evidence_path=str(path),
        plan_id=evidence.get("plan_id"),
        workload_id=(_identity(plan["workload"]) if plan is not None else None),
        environment_id=_environment_identity(evidence),
        hardware_cohort_id=_hardware_cohort_identity(evidence),
        committed_work_per_second=float(performance.get("committed_work_per_second", 0)),
        optimizer_steps=int(performance.get("optimizer_steps", 0)),
        peak_resident_bytes=int(lifecycle.get("peak_resident_bytes", -1)),
        planned_sample_pool_size=int(sample_pool_size),
        completed=evidence.get("state") == "completed" and evidence.get("health") == "healthy",
        finite_training_signals=finite_training,
        lifecycle_healthy=(
            cleanup.get("state") == "completed"
            and lifecycle.get("state") == "validated"
            and lifecycle.get("resident_outputs") == 0
            and lifecycle.get("resident_bytes") == 0
        ),
        required_capabilities_active=required_active and performance_active,
    )


@dataclass(frozen=True)
class CandidateAggregate:
    candidate_id: str
    repeats: int
    median_committed_work_per_second: float
    minimum_committed_work_per_second: float
    maximum_committed_work_per_second: float
    peak_resident_bytes: int
    planned_sample_pool_size: int


def select_candidate(
    measurements: Iterable[CandidateMeasurement], *, equivalence_fraction: float = 0.02
) -> dict[str, Any]:
    """Select a healthy candidate, preferring lower transfer residency at parity."""
    if not 0 <= equivalence_fraction < 1:
        raise ValueError("equivalence_fraction must be in [0, 1)")
    values = tuple(measurements)
    valid = [row for row in values if row.valid]
    if not valid:
        raise ValueError("no valid candidate measurements")
    grouped: defaultdict[str, list[CandidateMeasurement]] = defaultdict(list)
    for row in valid:
        grouped[row.candidate_id].append(row)
    aggregates = [
        CandidateAggregate(
            candidate_id=candidate_id,
            repeats=len(rows),
            median_committed_work_per_second=statistics.median(
                row.committed_work_per_second for row in rows
            ),
            minimum_committed_work_per_second=min(
                row.committed_work_per_second for row in rows
            ),
            maximum_committed_work_per_second=max(
                row.committed_work_per_second for row in rows
            ),
            peak_resident_bytes=max(row.peak_resident_bytes for row in rows),
            planned_sample_pool_size=max(
                row.planned_sample_pool_size for row in rows
            ),
        )
        for candidate_id, rows in grouped.items()
    ]
    fastest = max(row.median_committed_work_per_second for row in aggregates)
    floor = fastest * (1.0 - equivalence_fraction)
    equivalent = [
        row
        for row in aggregates
        if row.median_committed_work_per_second >= floor
    ]
    ranked = sorted(
        equivalent,
        key=lambda row: (
            row.peak_resident_bytes,
            row.planned_sample_pool_size,
            -row.median_committed_work_per_second,
            row.candidate_id,
        ),
    )
    return {
        "recommended": asdict(ranked[0]),
        "fastest_committed_work_per_second": fastest,
        "equivalent_floor_committed_work_per_second": floor,
        "ranked_equivalent": [asdict(row) for row in ranked],
        "rejected_measurements": len(values) - len(valid),
    }


def build_calibration_report(
    evidence_paths: Sequence[str | Path], *, equivalence_fraction: float = 0.02
) -> dict[str, Any]:
    measurements = [load_candidate_measurement(path) for path in evidence_paths]
    if not measurements:
        raise ValueError("at least one evidence path is required")
    workload_ids = {row.workload_id for row in measurements}
    if len(workload_ids) != 1 or None in workload_ids:
        raise ValueError("calibration evidence must describe exactly one workload")
    hardware_cohort_ids = {row.hardware_cohort_id for row in measurements}
    if len(hardware_cohort_ids) != 1:
        raise ValueError("calibration evidence must come from one hardware cohort")
    environment_ids = {row.environment_id for row in measurements}
    if len(environment_ids) != 1:
        raise ValueError("calibration evidence must come from exactly one environment")
    return {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "workload_id": measurements[0].workload_id,
        "environment_id": measurements[0].environment_id,
        "hardware_cohort_id": measurements[0].hardware_cohort_id,
        "measurement_count": len(measurements),
        "valid_measurement_count": sum(row.valid for row in measurements),
        "selection": select_candidate(
            measurements, equivalence_fraction=equivalence_fraction
        ),
        "measurements": [row.as_dict() for row in measurements],
    }
