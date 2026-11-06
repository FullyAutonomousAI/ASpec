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

"""Expert-bounded, service-rate-driven single-node topology planning.

Experts define coherent capability bundles and their legal parameter domains.
ASpec then derives role probes, measures producer and consumer service
capacity on the real path, predicts balanced integer allocations, and leaves
the final decision to bounded end-to-end runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from aspec.calibration import (
    candidate_identity_from_plan,
    load_candidate_measurement,
)
from aspec.config.draft_execution import is_direct_cp_backend
from aspec.config.train_config import load_config
from aspec.inference.concurrency import direct_output_sample_capacity
from aspec.planning import ExecutionPlan, compile_execution_plan

TOPOLOGY_STUDY_SCHEMA_VERSION = "aspec.topology-study/v0alpha2"
SERVICE_PROFILE_SCHEMA_VERSION = "aspec.service-profile/v0alpha1"
TOPOLOGY_RANKING_SCHEMA_VERSION = "aspec.topology-ranking/v0alpha1"
DEFAULT_TARGET_TENSOR_PARALLEL_SIZES = (1, 2, 4)
TARGET_TP_SELECTION_POLICY = "ordered_target_capacity_fallback"


def _identity(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _bool_override(value: bool) -> str:
    return str(bool(value)).lower()


def _select(config: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(config, key, default=default)


@dataclass(frozen=True)
class CapabilityBundle:
    """One coherent implementation path admitted by framework experts.

    The bundle says which mechanisms may be combined and which small domains
    are worth calibrating.  It deliberately does not prescribe a role split.
    """

    name: str
    attention_backend: str
    context_parallel_size: int
    data_plane: str
    transport_protocol: str
    target_output: str
    target_tensor_parallel_sizes: tuple[int, ...]
    inference_batch_sizes: tuple[int, ...]
    max_concurrent_batches: tuple[int, ...]
    requires_nvlink: bool
    gpu_direct: bool
    host_fallback_allowed: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("capability bundle name cannot be empty")
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive")
        if self.target_output not in {"buffered", "direct"}:
            raise ValueError("target_output must be 'buffered' or 'direct'")
        domains = {
            "target_tensor_parallel_sizes": self.target_tensor_parallel_sizes,
            "inference_batch_sizes": self.inference_batch_sizes,
            "max_concurrent_batches": self.max_concurrent_batches,
        }
        invalid = {
            name: values
            for name, values in domains.items()
            if not values or any(not isinstance(value, int) or value < 1 for value in values)
        }
        if invalid:
            raise ValueError(f"bundle domains must contain positive integers: {invalid}")
        if self.requires_nvlink and self.transport_protocol != "nvlink_intra":
            raise ValueError("requires_nvlink bundles must use protocol=nvlink_intra")
        if self.gpu_direct and self.data_plane != "transfer_engine":
            raise ValueError("gpu_direct requires data_plane=transfer_engine")

    @property
    def bundle_id(self) -> str:
        return _identity(asdict(self))


def current_capability_bundle(
    plan: ExecutionPlan,
    *,
    target_tensor_parallel_sizes: Sequence[int] | None = None,
) -> CapabilityBundle:
    """Turn the admitted path into a bundle with a small expert TP ladder.

    Target TP is an execution choice within the admitted SGLang path, not a
    separate optimization capability.  Values are ordered: the runner starts
    with the first TP and only advances after explicit Target-capacity evidence.
    """

    policy = plan.policy
    training = policy.parallelism["training"]
    inference = policy.parallelism["inference"]
    transfer = policy.capacity["transfer_storage"]
    pipeline = policy.pipeline
    direct = pipeline.get("target_feature_output") == "direct_to_transport"
    protocol = str(transfer.get("protocol", "tcp"))
    data_plane = str(transfer.get("data_plane", "store"))
    total_gpus = int(plan.policy.resources["training"]["world_size"]) + int(
        plan.policy.resources["inference"]["gpus"]
    )
    current_tp = int(inference["tensor_parallel"])
    if target_tensor_parallel_sizes is not None:
        tp_domain = tuple(dict.fromkeys(target_tensor_parallel_sizes))
    else:
        tp_domain = tuple(
            sorted(
                {
                    current_tp,
                    *(
                        value
                        for value in DEFAULT_TARGET_TENSOR_PARALLEL_SIZES
                        if value < total_gpus
                    ),
                }
            )
        )
    # The historical flag controls registered CUDA rings even for TCP, which
    # deliberately has no network GPUDirect capability claim.
    gpu_direct = bool(transfer.get("registered_cuda_buffers_required", False))
    return CapabilityBundle(
        name="current",
        attention_backend=str(training["attention_backend"]),
        context_parallel_size=int(training.get("context_parallel", 1)),
        data_plane=data_plane,
        transport_protocol=protocol,
        target_output="direct" if direct else "buffered",
        target_tensor_parallel_sizes=tp_domain,
        inference_batch_sizes=(int(pipeline["request_batch_size"]),),
        max_concurrent_batches=(int(pipeline["max_concurrent_batches"]),),
        requires_nvlink=protocol == "nvlink_intra" and not transfer.get(
            "host_fallback_allowed", True
        ),
        gpu_direct=gpu_direct,
        host_fallback_allowed=bool(transfer.get("host_fallback_allowed", True)),
    )


def first_class_capability_bundle(
    name: str, *, request_batch_size: int, max_concurrent_batches: int
) -> CapabilityBundle:
    """Resolve one built-in bundle whose implementation has product evidence."""

    common = {
        "data_plane": "transfer_engine",
        "transport_protocol": "nvlink_intra",
        "inference_batch_sizes": (int(request_batch_size),),
        "max_concurrent_batches": (int(max_concurrent_batches),),
        "requires_nvlink": True,
        "gpu_direct": True,
        "host_fallback_allowed": False,
        "target_tensor_parallel_sizes": DEFAULT_TARGET_TENSOR_PARALLEL_SIZES,
    }
    if name == "flex_cp1_buffered_nvlink":
        return CapabilityBundle(
            name=name,
            attention_backend="flex_attention",
            context_parallel_size=1,
            target_output="buffered",
            **common,
        )
    if name == "fa4_cp1_direct_nvlink":
        return CapabilityBundle(
            name=name,
            attention_backend="fa4",
            context_parallel_size=1,
            target_output="direct",
            **common,
        )
    if name == "fa4_cp2_direct_nvlink":
        return CapabilityBundle(
            name=name,
            attention_backend="fa4",
            context_parallel_size=2,
            target_output="direct",
            **common,
        )
    raise ValueError(f"unknown first-class capability bundle: {name}")


@dataclass(frozen=True)
class TopologyPolicy:
    """Bounded automatic-search policy, separate from capability legality."""

    buffer_headroom_dispatches: tuple[int, ...] = (0, 1)
    profile_headroom_dispatches: int = 1
    max_finalist_shapes: int = 2
    producer_headroom_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not self.buffer_headroom_dispatches:
            raise ValueError("buffer_headroom_dispatches cannot be empty")
        if any(value < 0 for value in self.buffer_headroom_dispatches):
            raise ValueError("buffer headroom must be non-negative")
        if self.profile_headroom_dispatches < 0:
            raise ValueError("profile headroom must be non-negative")
        if self.max_finalist_shapes < 1:
            raise ValueError("max_finalist_shapes must be positive")
        if not math.isfinite(self.producer_headroom_fraction) or self.producer_headroom_fraction < 0:
            raise ValueError("producer_headroom_fraction must be finite and non-negative")


@dataclass(frozen=True)
class CompiledTopologyCandidate:
    candidate_id: str
    plan_id: str
    bundle_id: str
    bundle_name: str
    total_gpus: int
    trainer_gpus: int
    target_gpus: int
    target_tensor_parallel_size: int
    target_engines: int
    context_parallel_size: int
    logical_data_parallel_size: int
    micro_batch_size: int
    accumulation_steps: int
    global_batch_size: int
    inference_batch_size: int
    max_concurrent_batches: int
    sample_pool_size: int
    pool_headroom_dispatches: int
    overrides: tuple[str, ...]
    plan: Mapping[str, Any]

    @property
    def shape_id(self) -> str:
        return _identity(
            {
                "bundle_id": self.bundle_id,
                "trainer_gpus": self.trainer_gpus,
                "target_gpus": self.target_gpus,
                "target_tensor_parallel_size": self.target_tensor_parallel_size,
                "micro_batch_size": self.micro_batch_size,
                "inference_batch_size": self.inference_batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
            }
        )

    @property
    def label(self) -> str:
        return (
            f"{self.bundle_name}-target{self.target_gpus}-tp{self.target_tensor_parallel_size}"
            f"-trainer{self.trainer_gpus}-b{self.inference_batch_size}"
            f"-c{self.max_concurrent_batches}-pool{self.sample_pool_size}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shape_id": self.shape_id, "label": self.label}


@dataclass(frozen=True)
class TopologyStudy:
    session_id: str
    base_plan_id: str
    workload_id: str
    total_gpus: int
    target_tp_selection_policy: str
    bundles: tuple[CapabilityBundle, ...]
    policy: TopologyPolicy
    candidates: tuple[CompiledTopologyCandidate, ...]
    profile_candidate_ids: tuple[str, ...]
    rejection_counts: Mapping[str, int]

    def candidate(self, candidate_id: str) -> CompiledTopologyCandidate:
        try:
            return next(row for row in self.candidates if row.candidate_id == candidate_id)
        except StopIteration:
            raise ValueError(f"candidate is not part of topology session: {candidate_id}") from None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TOPOLOGY_STUDY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "base_plan_id": self.base_plan_id,
            "workload_id": self.workload_id,
            "total_gpus": self.total_gpus,
            "target_tp_selection_policy": self.target_tp_selection_policy,
            "bundles": [{**asdict(row), "bundle_id": row.bundle_id} for row in self.bundles],
            "policy": asdict(self.policy),
            "candidate_count": len(self.candidates),
            "profile_candidate_ids": list(self.profile_candidate_ids),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _candidate_overrides(
    base_config: DictConfig,
    base_plan: ExecutionPlan,
    bundle: CapabilityBundle,
    *,
    total_gpus: int,
    trainer_gpus: int,
    target_tensor_parallel_size: int,
    inference_batch_size: int,
    max_concurrent_batches: int,
    pool_headroom_dispatches: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    target_gpus = total_gpus - trainer_gpus
    cp_size = bundle.context_parallel_size
    micro_batch_size = int(base_plan.policy.pipeline["train_micro_batch_size"])
    global_batch_size = int(base_plan.policy.pipeline["global_batch_size"])
    logical_dp = trainer_gpus // cp_size
    dispatch_width = logical_dp * micro_batch_size
    accumulation_steps = global_batch_size // dispatch_width
    target_engines = target_gpus // target_tensor_parallel_size
    packing = bool(base_plan.policy.pipeline["packed_rows"])
    packing_minimum = (
        int(base_plan.policy.pipeline.get("packing_min_samples_per_dispatch", 0))
        if packing
        else 0
    )
    dispatch_source_minimum = max(dispatch_width, packing_minimum)
    prefetch_depth = int(base_plan.policy.pipeline.get("prefetch_depth", 0))
    consumer_window = dispatch_source_minimum * (1 + int(prefetch_depth > 0))
    producer_window = target_engines * inference_batch_size * max_concurrent_batches
    structural_pool = max(dispatch_source_minimum, consumer_window, producer_window)
    sample_pool_size = structural_pool + pool_headroom_dispatches * dispatch_width

    transfer = base_plan.policy.capacity["transfer_storage"]
    common_slots = int(transfer.get("gpu_ring_slots") or 2)
    source_slots = transfer.get("source_gpu_ring_slots")
    destination_slots = transfer.get("destination_gpu_ring_slots")
    source_slots = common_slots if source_slots is None else int(source_slots)
    destination_slots = common_slots if destination_slots is None else int(destination_slots)
    slots_per_sample = cp_size if is_direct_cp_backend(bundle.attention_backend) else 1
    source_slots = max(
        source_slots,
        inference_batch_size * slots_per_sample * max_concurrent_batches,
    )
    if cp_size > 1:
        destination_slots = max(destination_slots, micro_batch_size)

    overrides = (
        f"training.attention_backend={bundle.attention_backend}",
        f"training.draft_context_parallel_size={cp_size}",
        "training.streaming_dispatch_with_training=true",
        f"training.training_num_gpus_per_node={trainer_gpus}",
        f"training.micro_batch_size={micro_batch_size}",
        f"training.draft_accumulation_steps={accumulation_steps}",
        f"training.max_concurrent_batches={max_concurrent_batches}",
        f"inference.inference_num_gpus={target_gpus}",
        f"inference.inference_num_gpus_per_node={target_gpus}",
        f"inference.inference_num_gpus_per_engine={target_tensor_parallel_size}",
        f"inference.inference_batch_size={inference_batch_size}",
        f"inference.inference_fetch_batch={sample_pool_size}",
        f"inference.inference_buffer_threshold={sample_pool_size}",
        f"inference.max_sample_pool_size={sample_pool_size}",
        f"inference.target_prefill_memory_optimized={_bool_override(bundle.target_output == 'direct')}",
        f"inference.sglang.tp_size={target_tensor_parallel_size}",
        f"mooncake.data_plane={bundle.data_plane}",
        f"mooncake.protocol={bundle.transport_protocol}",
        f"mooncake.require_nvlink={_bool_override(bundle.requires_nvlink)}",
        f"mooncake.enable_gpu_direct={_bool_override(bundle.gpu_direct)}",
        f"mooncake.allow_host_fallback={_bool_override(bundle.host_fallback_allowed)}",
        f"mooncake.source_gpu_ring_slots={source_slots}",
        f"mooncake.destination_gpu_ring_slots={destination_slots}",
    )
    return overrides, {
        "target_gpus": target_gpus,
        "target_engines": target_engines,
        "logical_dp": logical_dp,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "global_batch_size": global_batch_size,
        "sample_pool_size": sample_pool_size,
    }


def compile_topology_study(
    base_config: DictConfig,
    base_plan: ExecutionPlan,
    *,
    total_gpus: int,
    bundles: Sequence[CapabilityBundle] | None = None,
    policy: TopologyPolicy | None = None,
    config_path: str | None = None,
) -> TopologyStudy:
    """Compile every legal integer allocation and choose role-profile probes."""

    if total_gpus < 2:
        raise ValueError("automatic topology planning requires at least two GPUs")
    bundles = tuple(bundles or (current_capability_bundle(base_plan),))
    if not bundles:
        raise ValueError("at least one capability bundle is required")
    policy = policy or TopologyPolicy()
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise ValueError("capability bundles must be unique")

    candidates: dict[str, CompiledTopologyCandidate] = {}
    rejected: Counter[str] = Counter()
    base_workload = asdict(base_plan.workload)

    for bundle in bundles:
        cp_size = bundle.context_parallel_size
        micro_batch_size = int(base_plan.policy.pipeline["train_micro_batch_size"])
        global_batch_size = int(base_plan.policy.pipeline["global_batch_size"])
        for target_tp in bundle.target_tensor_parallel_sizes:
            for trainer_gpus in range(cp_size, total_gpus):
                target_gpus = total_gpus - trainer_gpus
                if trainer_gpus % cp_size:
                    rejected["trainer_not_divisible_by_context_parallel"] += 1
                    continue
                if target_gpus < target_tp or target_gpus % target_tp:
                    rejected["target_not_divisible_by_tensor_parallel"] += 1
                    continue
                logical_dp = trainer_gpus // cp_size
                dispatch_width = logical_dp * micro_batch_size
                if global_batch_size % dispatch_width:
                    rejected["global_batch_not_divisible_by_dispatch"] += 1
                    continue
                target_engines = target_gpus // target_tp
                transfer_capacity = base_plan.policy.capacity["transfer_storage"]
                source_slots = transfer_capacity.get("source_gpu_ring_slots")
                if source_slots is None:
                    source_slots = transfer_capacity.get("gpu_ring_slots") or 2
                slots_per_sample = (
                    cp_size if is_direct_cp_backend(bundle.attention_backend) else 1
                )
                if bundle.target_output == "direct" and direct_output_sample_capacity(
                    engine_count=target_engines,
                    source_slots=int(source_slots),
                    slots_per_sample=slots_per_sample,
                ) < dispatch_width:
                    rejected["direct_output_source_ring_below_dispatch"] += 1
                    continue
                for batch_size in bundle.inference_batch_sizes:
                    for concurrency in bundle.max_concurrent_batches:
                        for headroom in policy.buffer_headroom_dispatches:
                            overrides, derived = _candidate_overrides(
                                base_config,
                                base_plan,
                                bundle,
                                total_gpus=total_gpus,
                                trainer_gpus=trainer_gpus,
                                target_tensor_parallel_size=target_tp,
                                inference_batch_size=batch_size,
                                max_concurrent_batches=concurrency,
                                pool_headroom_dispatches=headroom,
                            )
                            try:
                                config = load_config(
                                    base_config=base_config,
                                    cli_args=list(overrides),
                                )
                                plan = compile_execution_plan(
                                    config,
                                    config_path=config_path,
                                    cli_args=overrides,
                                )
                            except (ValueError, NotImplementedError) as exc:
                                reason = str(exc).split(":", 1)[0].strip()
                                rejected[f"canonical_compile: {reason}"] += 1
                                continue
                            if asdict(plan.workload) != base_workload:
                                raise RuntimeError(
                                    "topology candidate changed the canonical Workload"
                                )
                            plan_dict = plan.to_dict()
                            candidate_id = candidate_identity_from_plan(plan_dict)
                            candidates[candidate_id] = CompiledTopologyCandidate(
                                candidate_id=candidate_id,
                                plan_id=plan.plan_id,
                                bundle_id=bundle.bundle_id,
                                bundle_name=bundle.name,
                                total_gpus=total_gpus,
                                trainer_gpus=trainer_gpus,
                                target_gpus=derived["target_gpus"],
                                target_tensor_parallel_size=target_tp,
                                target_engines=derived["target_engines"],
                                context_parallel_size=cp_size,
                                logical_data_parallel_size=derived["logical_dp"],
                                micro_batch_size=derived["micro_batch_size"],
                                accumulation_steps=derived["accumulation_steps"],
                                global_batch_size=derived["global_batch_size"],
                                inference_batch_size=batch_size,
                                max_concurrent_batches=concurrency,
                                sample_pool_size=derived["sample_pool_size"],
                                pool_headroom_dispatches=headroom,
                                overrides=overrides,
                                plan=plan_dict,
                            )

    bundle_order = {bundle.bundle_id: index for index, bundle in enumerate(bundles)}
    tp_order = {
        (bundle.bundle_id, target_tp): index
        for bundle in bundles
        for index, target_tp in enumerate(bundle.target_tensor_parallel_sizes)
    }
    ordered = tuple(
        sorted(
            candidates.values(),
            key=lambda row: (
                bundle_order[row.bundle_id],
                tp_order[(row.bundle_id, row.target_tensor_parallel_size)],
                row.inference_batch_size,
                row.max_concurrent_batches,
                row.trainer_gpus,
                row.pool_headroom_dispatches,
            ),
        )
    )
    if not ordered:
        raise ValueError(f"expert capability domain produced no legal candidates: {dict(rejected)}")

    profile_ids: list[str] = []
    groups: dict[tuple[str, int, int, int], list[CompiledTopologyCandidate]] = {}
    for candidate in ordered:
        if candidate.pool_headroom_dispatches != policy.profile_headroom_dispatches:
            continue
        key = (
            candidate.bundle_id,
            candidate.target_tensor_parallel_size,
            candidate.inference_batch_size,
            candidate.max_concurrent_batches,
        )
        groups.setdefault(key, []).append(candidate)
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: row.trainer_gpus)
        profile_ids.append(rows[0].candidate_id)
        if rows[-1].candidate_id != rows[0].candidate_id:
            profile_ids.append(rows[-1].candidate_id)
    profile_ids = list(dict.fromkeys(profile_ids))
    if not profile_ids:
        raise ValueError(
            "profile_headroom_dispatches does not match any compiled buffer candidate"
        )

    workload_id = _identity(base_workload)
    session_payload = {
        "base_plan_id": base_plan.plan_id,
        "workload_id": workload_id,
        "total_gpus": total_gpus,
        "target_tp_selection_policy": TARGET_TP_SELECTION_POLICY,
        "bundles": [bundle.bundle_id for bundle in bundles],
        "policy": asdict(policy),
        "candidates": [candidate.candidate_id for candidate in ordered],
    }
    return TopologyStudy(
        session_id=_identity(session_payload),
        base_plan_id=base_plan.plan_id,
        workload_id=workload_id,
        total_gpus=total_gpus,
        target_tp_selection_policy=TARGET_TP_SELECTION_POLICY,
        bundles=tuple(bundles),
        policy=policy,
        candidates=ordered,
        profile_candidate_ids=tuple(profile_ids),
        rejection_counts=dict(rejected),
    )


@dataclass(frozen=True)
class ServiceObservation:
    session_id: str
    candidate_id: str
    evidence_path: str
    environment_id: str
    bundle_id: str
    target_tensor_parallel_size: int
    target_engines: int
    inference_batch_size: int
    max_concurrent_batches: int
    context_parallel_size: int
    logical_data_parallel_size: int
    micro_batch_size: int
    producer_work_per_second: float
    consumer_work_per_second: float
    end_to_end_work_per_second: float
    committed_work_per_training_row: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SERVICE_PROFILE_SCHEMA_VERSION, **asdict(self)}


def load_service_observation(
    study: TopologyStudy, evidence_path: str | Path
) -> ServiceObservation:
    """Load role service rates from one healthy, session-bound real run."""

    measurement = load_candidate_measurement(evidence_path)
    if not measurement.valid:
        raise ValueError(f"service profile run is not healthy: {evidence_path}")
    candidate = study.candidate(measurement.candidate_id)
    path = Path(evidence_path).expanduser().resolve()
    evidence = json.loads(path.read_text(encoding="utf-8"))
    training = evidence["observations"]["training"]
    service_profiles = training.get("service_profiles") or {}
    producer = service_profiles.get("producer") or {}
    consumer = service_profiles.get("consumer") or {}
    if producer.get("state") != "measured" or consumer.get("state") != "measured":
        raise ValueError(
            "service profile requires independently measured producer and consumer rates"
        )
    producer_capacity = producer.get("work_per_second")
    consumer_capacity = consumer.get("work_per_second")
    performance = training.get("performance") or {}
    values = (producer_capacity, consumer_capacity)
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError(
            "service profile requires finite positive producer and consumer work rates"
        )
    measured_rows = performance["optimizer_steps"] * candidate.global_batch_size
    work_factor = performance["committed_work"] / measured_rows
    if not math.isfinite(work_factor) or work_factor <= 0:
        raise ValueError("service profile has an invalid committed-work conversion")
    return ServiceObservation(
        session_id=study.session_id,
        candidate_id=candidate.candidate_id,
        evidence_path=str(path),
        environment_id=measurement.environment_id,
        bundle_id=candidate.bundle_id,
        target_tensor_parallel_size=candidate.target_tensor_parallel_size,
        target_engines=candidate.target_engines,
        inference_batch_size=candidate.inference_batch_size,
        max_concurrent_batches=candidate.max_concurrent_batches,
        context_parallel_size=candidate.context_parallel_size,
        logical_data_parallel_size=candidate.logical_data_parallel_size,
        micro_batch_size=candidate.micro_batch_size,
        producer_work_per_second=float(producer_capacity),
        consumer_work_per_second=float(consumer_capacity),
        end_to_end_work_per_second=measurement.committed_work_per_second,
        committed_work_per_training_row=work_factor,
    )


@dataclass(frozen=True)
class RankedTopologyCandidate:
    candidate_id: str
    shape_id: str
    predicted_producer_work_per_second: float
    predicted_consumer_work_per_second: float
    predicted_end_to_end_work_per_second: float
    producer_to_consumer_ratio: float
    model_efficiency: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _median_unit_rate(
    values: Sequence[ServiceObservation], *, capacity_field: str, units_field: str
) -> float:
    rates = [
        float(getattr(row, capacity_field)) / int(getattr(row, units_field))
        for row in values
    ]
    return float(statistics.median(rates))


def rank_topology_candidates(
    study: TopologyStudy,
    observations: Sequence[ServiceObservation],
) -> dict[str, Any]:
    """Predict balanced allocations and return a small end-to-end frontier."""

    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one service observation is required")
    if any(row.session_id != study.session_id for row in observations):
        raise ValueError("service observations belong to another topology session")
    environment_ids = {row.environment_id for row in observations}
    if len(environment_ids) != 1:
        raise ValueError("service observations must come from one comparable environment")

    ranked: list[RankedTopologyCandidate] = []
    omitted: Counter[str] = Counter()
    for candidate in study.candidates:
        producer_profiles = [
            row
            for row in observations
            if row.bundle_id == candidate.bundle_id
            and row.target_tensor_parallel_size
            == candidate.target_tensor_parallel_size
            and row.inference_batch_size == candidate.inference_batch_size
            and row.max_concurrent_batches == candidate.max_concurrent_batches
        ]
        consumer_profiles = [
            row
            for row in observations
            if row.bundle_id == candidate.bundle_id
            and row.context_parallel_size == candidate.context_parallel_size
            and row.micro_batch_size == candidate.micro_batch_size
        ]
        if not producer_profiles:
            omitted["missing_producer_profile"] += 1
            continue
        if not consumer_profiles:
            omitted["missing_consumer_profile"] += 1
            continue
        producer_rate = _median_unit_rate(
            producer_profiles,
            capacity_field="producer_work_per_second",
            units_field="target_engines",
        )
        consumer_rate = _median_unit_rate(
            consumer_profiles,
            capacity_field="consumer_work_per_second",
            units_field="logical_data_parallel_size",
        )
        predicted_producer = producer_rate * candidate.target_engines
        predicted_consumer = consumer_rate * candidate.logical_data_parallel_size
        efficiencies = []
        for row in observations:
            if row not in producer_profiles or row not in consumer_profiles:
                continue
            denominator = min(
                row.producer_work_per_second,
                row.consumer_work_per_second,
            )
            efficiencies.append(min(1.0, row.end_to_end_work_per_second / denominator))
        efficiency = float(statistics.median(efficiencies)) if efficiencies else 1.0
        predicted_end_to_end = min(predicted_producer, predicted_consumer) * efficiency
        ranked.append(
            RankedTopologyCandidate(
                candidate_id=candidate.candidate_id,
                shape_id=candidate.shape_id,
                predicted_producer_work_per_second=predicted_producer,
                predicted_consumer_work_per_second=predicted_consumer,
                predicted_end_to_end_work_per_second=predicted_end_to_end,
                producer_to_consumer_ratio=predicted_producer / predicted_consumer,
                model_efficiency=efficiency,
            )
        )

    target_ratio = 1.0 + study.policy.producer_headroom_fraction
    ranked.sort(
        key=lambda row: (
            -row.predicted_end_to_end_work_per_second,
            abs(math.log(row.producer_to_consumer_ratio / target_ratio)),
            row.candidate_id,
        )
    )
    if not ranked:
        raise ValueError(f"no candidate has complete role profiles: {dict(omitted)}")

    selected_shapes: list[str] = []
    for row in ranked:
        if row.shape_id not in selected_shapes:
            selected_shapes.append(row.shape_id)
        if len(selected_shapes) == study.policy.max_finalist_shapes:
            break
    finalist_ids = [
        row.candidate_id for row in ranked if row.shape_id in selected_shapes
    ]
    return {
        "schema_version": TOPOLOGY_RANKING_SCHEMA_VERSION,
        "session_id": study.session_id,
        "environment_id": next(iter(environment_ids)),
        "service_observations": [row.to_dict() for row in observations],
        "ranked_candidates": [row.to_dict() for row in ranked],
        "finalist_candidate_ids": finalist_ids,
        "omitted_candidate_counts": dict(sorted(omitted.items())),
        "model_scope": {
            "kind": "linear_replica_service_rate_prior",
            "final_decision_requires_end_to_end_evidence": True,
        },
    }
