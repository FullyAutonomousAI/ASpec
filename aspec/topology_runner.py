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

"""Budgeted subprocess runner for automatic single-node topology calibration."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from aspec.calibration import (
    CandidateMeasurement,
    candidate_identity_from_plan,
    load_candidate_measurement,
    select_candidate,
)
from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan
from aspec.topology import (
    CapabilityBundle,
    CompiledTopologyCandidate,
    TopologyPolicy,
    compile_topology_study,
    load_service_observation,
    rank_topology_candidates,
)

AUTOMATIC_TOPOLOGY_RUN_SCHEMA_VERSION = "aspec.automatic-topology-run/v0alpha3"

_TARGET_ENGINE_MARKERS = (
    "ray::sglengine.init",
    "class_name: sglengine",
    "aspec/inference/engine/sgl_engine.py",
    "aspec.inference.engine.sgl_engine",
)
_TARGET_CAPACITY_MARKERS = {
    "profiled_token_capacity": (
        "sglang profiled max_total_tokens cannot satisfy",
        "requested target-prefill concurrency",
    ),
    "kv_cache_capacity": (
        "kv cache pool is too small",
        "not enough kv cache",
        "insufficient kv cache",
    ),
    "target_memory_capacity": (
        "torch.outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "not enough memory",
        "insufficient memory",
        "cannot allocate memory",
        "failed to allocate memory",
    ),
}

_PROTOCOL_POLICY_FIELDS = (
    "confirm_repeats",
    "equivalence_fraction",
    "calibration_steps",
    "calibration_warmup_steps",
    "calibration_drain_steps",
    "producer_profile_samples",
    "producer_profile_warmup_samples",
)
_OPERATIONAL_POLICY_FIELDS = (
    "total_budget_seconds",
    "per_run_timeout_seconds",
    "post_run_quiescence_timeout_seconds",
)


@dataclass(frozen=True)
class AutomaticRunPolicy:
    total_budget_seconds: float = 7200.0
    per_run_timeout_seconds: float = 1800.0
    confirm_repeats: int = 1
    equivalence_fraction: float = 0.02
    calibration_steps: int = 3
    calibration_warmup_steps: int = 1
    calibration_drain_steps: int = 0
    producer_profile_samples: int = 64
    producer_profile_warmup_samples: int = 16
    post_run_quiescence_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.total_budget_seconds <= 0:
            raise ValueError("total_budget_seconds must be positive")
        if self.per_run_timeout_seconds <= 0:
            raise ValueError("per_run_timeout_seconds must be positive")
        if self.confirm_repeats < 0:
            raise ValueError("confirm_repeats must be non-negative")
        if not 0 <= self.equivalence_fraction < 1:
            raise ValueError("equivalence_fraction must be in [0, 1)")
        if self.calibration_steps <= (
            self.calibration_warmup_steps + self.calibration_drain_steps
        ):
            raise ValueError(
                "calibration_steps must exceed warmup_steps + drain_steps"
            )
        if self.calibration_warmup_steps < 0 or self.calibration_drain_steps < 0:
            raise ValueError("calibration warmup/drain steps must be non-negative")
        if self.producer_profile_samples <= 0:
            raise ValueError("producer_profile_samples must be positive")
        if not 0 <= self.producer_profile_warmup_samples < self.producer_profile_samples:
            raise ValueError(
                "producer_profile_warmup_samples must be non-negative and smaller "
                "than producer_profile_samples"
            )
        if self.post_run_quiescence_timeout_seconds <= 0:
            raise ValueError("post_run_quiescence_timeout_seconds must be positive")


def _run_command(command: Sequence[str], *, timeout: float = 30.0) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed.stdout


def discover_hardware(*, hardware_label: str | None = None) -> dict[str, Any]:
    """Record physical facts needed to audit one single-node study."""

    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, name, memory_mib, compute_capability, driver = fields
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "reported_name": name,
                "memory_mib": int(memory_mib),
                "compute_capability": compute_capability,
                "driver_version": driver,
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi reported no visible GPUs")
    try:
        topology_matrix = _run_command(["nvidia-smi", "topo", "-m"])
    except RuntimeError as exc:
        topology_matrix = f"unavailable: {exc}"
    return {
        "hardware_label": hardware_label,
        "visible_gpu_count": len(gpus),
        "gpus": gpus,
        "topology_matrix": topology_matrix,
    }


def _identity(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _gpu_peer_links(topology_matrix: str, gpu_count: int) -> list[list[str]]:
    links: dict[int, list[str]] = {}
    for line in topology_matrix.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("GPU"):
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit() or int(suffix) >= gpu_count:
            continue
        row = fields[1 : 1 + gpu_count]
        index = int(suffix)
        if len(row) != gpu_count or row[index] != "X":
            continue
        links[index] = row
    return [links[index] for index in range(gpu_count)] if len(links) == gpu_count else []


def hardware_cohort(hardware: dict[str, Any]) -> dict[str, Any]:
    """Normalize host-specific discovery into a comparable hardware cohort."""

    gpus = hardware["gpus"]
    hardware_label = hardware.get("hardware_label")
    links = _gpu_peer_links(hardware.get("topology_matrix", ""), len(gpus))
    peer_links = sorted(
        links[source][destination]
        for source in range(len(links))
        for destination in range(source + 1, len(links))
    )
    payload = {
        "gpu_model": hardware_label or gpus[0]["reported_name"],
        "visible_gpu_count": len(gpus),
        "gpu_facts": sorted(
            (
                {
                    "memory_mib": gpu["memory_mib"],
                    "compute_capability": gpu["compute_capability"],
                }
                for gpu in gpus
            ),
            key=lambda row: json.dumps(row, sort_keys=True),
        ),
        "peer_links": peer_links,
    }
    return {**payload, "cohort_id": _identity(payload)}


def _validate_hardware_for_study(
    hardware: dict[str, Any], bundles: Sequence[CapabilityBundle]
) -> None:
    signatures = {
        (
            gpu["reported_name"],
            gpu["memory_mib"],
            gpu["compute_capability"],
        )
        for gpu in hardware["gpus"]
    }
    if len(signatures) != 1:
        raise ValueError(
            "automatic topology planning requires homogeneous visible GPUs; "
            f"observed {sorted(signatures)}"
        )
    if not any(bundle.requires_nvlink for bundle in bundles):
        return
    links = _gpu_peer_links(
        hardware.get("topology_matrix", ""), hardware["visible_gpu_count"]
    )
    if not links:
        raise ValueError("required NVLink topology could not be parsed from nvidia-smi")
    non_nvlink = [
        (source, destination, links[source][destination])
        for source in range(len(links))
        for destination in range(source + 1, len(links))
        if not links[source][destination].startswith("NV")
    ]
    if non_nvlink:
        raise ValueError(
            "required NVLink bundle needs all visible GPU pairs connected by NVLink; "
            f"non-NVLink pairs: {non_nvlink}"
        )


def _policy_fields(
    policy: AutomaticRunPolicy, fields: Sequence[str]
) -> dict[str, Any]:
    values = asdict(policy)
    return {field: values[field] for field in fields}


def _active_runtime_processes() -> dict[str, list[str]]:
    ray = subprocess.run(
        ["pgrep", "-af", "raylet|gcs_server|aspec.train_entry"],
        capture_output=True,
        check=False,
        text=True,
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "ray": [line for line in ray.stdout.splitlines() if line.strip()],
        "gpu": [line for line in gpu.stdout.splitlines() if line.strip()],
    }


def _wait_for_runtime_quiescence(
    *, timeout_seconds: float, poll_seconds: float = 0.5
) -> tuple[bool, dict[str, list[str]], float]:
    """Wait for a launched attempt's Ray processes and GPU contexts to disappear."""

    started = time.monotonic()
    deadline = started + timeout_seconds
    active = _active_runtime_processes()
    while active["ray"] or active["gpu"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, active, time.monotonic() - started
        time.sleep(min(poll_seconds, remaining))
        active = _active_runtime_processes()
    return True, active, time.monotonic() - started


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _failure_text_sources(attempt: dict[str, Any]) -> list[tuple[str, str]]:
    """Load bounded diagnostic text without treating generic failures as capacity."""

    sources: list[tuple[str, str]] = []
    evidence_path = attempt.get("evidence_path")
    if evidence_path:
        try:
            evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        else:
            error = evidence.get("error") or {}
            if isinstance(error, dict):
                text = "\n".join(
                    str(error.get(field) or "") for field in ("type", "message")
                )
                if text.strip():
                    sources.append(("evidence.error", text))
    run_dir = attempt.get("run_dir")
    if run_dir:
        log_path = Path(run_dir) / "train.log"
        try:
            with log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 2_000_000))
                text = stream.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        else:
            if text.strip():
                sources.append(("train.log", text))
    return sources


def _classify_role_profile_failure(attempt: dict[str, Any]) -> dict[str, Any]:
    """Recognize only explicit Target-engine capacity failures as TP fallbacks."""

    if attempt.get("post_run_quiescence_state") == "timed_out":
        return {
            "kind": "unclassified_failure",
            "signals": ["runtime_not_quiescent"],
            "source": None,
        }
    for source, text in _failure_text_sources(attempt):
        lowered = text.lower()
        signals: list[str] = []
        for name, markers in _TARGET_CAPACITY_MARKERS.items():
            for marker in markers:
                start = lowered.find(marker)
                while start >= 0:
                    if name == "profiled_token_capacity":
                        signals.append(name)
                        break
                    window = lowered[max(0, start - 4000) : start + 4000]
                    if any(role in window for role in _TARGET_ENGINE_MARKERS):
                        signals.append(name)
                        break
                    start = lowered.find(marker, start + len(marker))
                if name in signals:
                    break
        if signals:
            return {
                "kind": "target_capacity_infeasible",
                "signals": list(dict.fromkeys(signals)),
                "source": source,
            }
    return {
        "kind": "unclassified_failure",
        "signals": [],
        "source": None,
    }


class AutomaticTopologyRunner:
    """Run role probes, predicted finalists, and winner confirmation."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        output_root: str | Path,
        total_gpus: int,
        bundles: Sequence[CapabilityBundle] | None = None,
        topology_policy: TopologyPolicy | None = None,
        run_policy: AutomaticRunPolicy | None = None,
        hardware_label: str | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.total_gpus = int(total_gpus)
        self.topology_policy = topology_policy or TopologyPolicy()
        self.run_policy = run_policy or AutomaticRunPolicy()
        self.hardware = discover_hardware(hardware_label=hardware_label)
        if self.hardware["visible_gpu_count"] != self.total_gpus:
            raise ValueError(
                "total_gpus must equal the visible single-node GPU count: "
                f"{self.total_gpus} != {self.hardware['visible_gpu_count']}"
            )
        self.base_overrides = (
            f"training.num_train_steps={self.run_policy.calibration_steps}",
            "training.save_final_checkpoint=false",
            "debug.performance_measurement_required=true",
            "debug.performance_warmup_steps="
            f"{self.run_policy.calibration_warmup_steps}",
            "debug.performance_drain_steps="
            f"{self.run_policy.calibration_drain_steps}",
            "debug.topology_producer_profile_samples="
            f"{self.run_policy.producer_profile_samples}",
            "debug.topology_producer_profile_warmup_samples="
            f"{self.run_policy.producer_profile_warmup_samples}",
        )
        self.base_config = load_config(
            str(self.config_path), cli_args=list(self.base_overrides)
        )
        self.base_plan = compile_execution_plan(
            self.base_config,
            config_path=str(self.config_path),
            cli_args=self.base_overrides,
        )
        self.study = compile_topology_study(
            self.base_config,
            self.base_plan,
            total_gpus=self.total_gpus,
            bundles=bundles,
            policy=self.topology_policy,
            config_path=str(self.config_path),
        )
        _validate_hardware_for_study(self.hardware, self.study.bundles)
        self.hardware_cohort = hardware_cohort(self.hardware)
        self.formal_candidate_ids = {
            candidate.candidate_id: self._compile_formal_candidate_id(candidate)
            for candidate in self.study.candidates
        }
        self.state_path = self.output_root / "automatic-topology.json"
        self._started = time.monotonic()
        self._state = self._load_or_initialize_state()
        self._prior_run_seconds = sum(
            float(attempt.get("duration_seconds") or 0)
            for attempt in self._state["attempts"]
        )

    def _compile_formal_candidate_id(
        self, candidate: CompiledTopologyCandidate
    ) -> str:
        config = load_config(
            str(self.config_path), cli_args=list(candidate.overrides)
        )
        plan = compile_execution_plan(
            config,
            config_path=str(self.config_path),
            cli_args=candidate.overrides,
        )
        return candidate_identity_from_plan(plan.to_dict())

    def _load_or_initialize_state(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        entries = [path for path in self.output_root.iterdir() if path != self.state_path]
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != AUTOMATIC_TOPOLOGY_RUN_SCHEMA_VERSION:
                raise ValueError(
                    "automatic topology session uses an incompatible runner protocol; "
                    "choose a new output directory"
                )
            if state.get("session_id") != self.study.session_id:
                raise ValueError(
                    "output_root contains another topology session; choose a new directory"
                )
            stored_policy = state.get("run_policy") or {}
            stored_protocol = state.get("protocol_policy") or {
                field: stored_policy.get(field) for field in _PROTOCOL_POLICY_FIELDS
            }
            requested_protocol = _policy_fields(
                self.run_policy, _PROTOCOL_POLICY_FIELDS
            )
            if stored_protocol != requested_protocol:
                raise ValueError(
                    "resuming a topology session cannot change its measurement protocol: "
                    f"stored={stored_protocol}, requested={requested_protocol}"
                )
            stored_hardware_cohort = state.get("hardware_cohort")
            if stored_hardware_cohort is None:
                stored_hardware_cohort = hardware_cohort(state["hardware"])
            if (
                stored_hardware_cohort.get("cohort_id")
                != self.hardware_cohort["cohort_id"]
            ):
                raise ValueError(
                    "resuming a topology session requires the same hardware cohort: "
                    f"stored={stored_hardware_cohort.get('cohort_id')}, "
                    f"current={self.hardware_cohort['cohort_id']}"
                )
            stored_formal_ids = state.get("formal_candidate_ids")
            if (
                stored_formal_ids is not None
                and stored_formal_ids != self.formal_candidate_ids
            ):
                raise ValueError(
                    "formal execution semantics changed since the topology session began"
                )
            operational = _policy_fields(
                self.run_policy, _OPERATIONAL_POLICY_FIELDS
            )
            state.setdefault("protocol_policy", stored_protocol)
            state.setdefault("hardware_cohort", stored_hardware_cohort)
            state.setdefault("formal_candidate_ids", self.formal_candidate_ids)
            state.setdefault("resume_policy_history", []).append(
                {
                    "resumed_at_unix": time.time(),
                    "operational_policy": operational,
                }
            )
            state["active_operational_policy"] = operational
            _atomic_write(self.state_path, state)
            return state
        if entries:
            raise ValueError(
                "automatic topology output_root must be empty unless it contains a resumable session"
            )
        state = {
            "schema_version": AUTOMATIC_TOPOLOGY_RUN_SCHEMA_VERSION,
            "session_id": self.study.session_id,
            "state": "running",
            "started_at_unix": time.time(),
            "finished_at_unix": None,
            "config_path": str(self.config_path),
            "hardware": self.hardware,
            "hardware_cohort": self.hardware_cohort,
            "topology_study": self.study.to_dict(),
            "run_policy": asdict(self.run_policy),
            "protocol_policy": _policy_fields(
                self.run_policy, _PROTOCOL_POLICY_FIELDS
            ),
            "active_operational_policy": _policy_fields(
                self.run_policy, _OPERATIONAL_POLICY_FIELDS
            ),
            "resume_policy_history": [],
            "formal_candidate_ids": self.formal_candidate_ids,
            "attempts": [],
            "target_tp_resolution": [],
            "ranking": None,
            "selection": None,
            "stop_reason": None,
        }
        _atomic_write(self.state_path, state)
        return state

    def _save(self) -> None:
        _atomic_write(self.state_path, self._state)

    def _remaining_budget(self) -> float:
        return (
            self.run_policy.total_budget_seconds
            - self._prior_run_seconds
            - (time.monotonic() - self._started)
        )

    def _attempts_for(self, candidate_id: str) -> list[dict[str, Any]]:
        return [
            attempt
            for attempt in self._state["attempts"]
            if attempt["candidate_id"] == candidate_id
        ]

    def _phase_attempts(
        self, candidate_id: str, phases: set[str]
    ) -> list[dict[str, Any]]:
        return [
            attempt
            for attempt in self._attempts_for(candidate_id)
            if attempt["phase"] in phases
        ]

    def _should_run_phase(self, candidate_id: str, phase: str) -> bool:
        attempts = self._phase_attempts(candidate_id, {phase})
        if any(attempt["state"] == "completed" for attempt in attempts):
            return False
        return not attempts or all(
            attempt["state"] == "interrupted" for attempt in attempts
        )

    def _should_run_selection_candidate(self, candidate_id: str) -> bool:
        attempts = self._phase_attempts(
            candidate_id, {"role_profile", "short_e2e"}
        )
        if any(attempt["state"] == "completed" for attempt in attempts):
            return False
        return not attempts or all(
            attempt["state"] == "interrupted" for attempt in attempts
        )

    def _recover_orphaned_attempts(self) -> None:
        running = [
            attempt
            for attempt in self._state["attempts"]
            if attempt["state"] == "running"
        ]
        if not running:
            return
        active = _active_runtime_processes()
        if active["ray"] or active["gpu"]:
            raise RuntimeError(
                "cannot resume a topology session while its prior runtime may still "
                f"be active: {active}"
            )
        recovered_at = time.time()
        for attempt in running:
            attempt["recovery_detected_at_unix"] = recovered_at
            attempt["post_run_quiescence_state"] = "clean_on_resume"
            started_at = attempt.get("started_at_unix")
            log_path = Path(attempt["run_dir"]) / "train.log"
            if (
                attempt.get("duration_seconds") is None
                and isinstance(started_at, (int, float))
                and log_path.is_file()
            ):
                attempt["duration_seconds"] = max(
                    0.0, min(recovered_at, log_path.stat().st_mtime) - started_at
                )
                attempt["duration_recovery_source"] = "train_log_mtime"
            evidence_path = Path(attempt["run_dir"]) / "evidence.json"
            if not evidence_path.is_file():
                attempt["state"] = "interrupted"
                attempt["error"] = (
                    "runner ended before the attempt produced terminal Evidence; "
                    "no live Ray/GPU runtime was present when recovery began"
                )
                continue
            attempt["evidence_path"] = str(evidence_path)
            try:
                measurement = load_candidate_measurement(evidence_path)
                candidate = self.study.candidate(attempt["candidate_id"])
                calibration_workload = bool(
                    attempt.get("calibration_workload", True)
                )
                expected_candidate_id = attempt.get(
                    "expected_executed_candidate_id"
                ) or (
                    candidate.candidate_id
                    if calibration_workload
                    else self.formal_candidate_ids[candidate.candidate_id]
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                attempt["state"] = "invalid"
                attempt["error"] = (
                    "orphaned attempt has invalid terminal Evidence: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            attempt["executed_candidate_id"] = measurement.candidate_id
            if (
                measurement.candidate_id != expected_candidate_id
                or not measurement.valid
            ):
                attempt["state"] = "invalid"
                attempt["error"] = (
                    "orphaned attempt terminal Evidence does not match the candidate "
                    "or did not pass health gates"
                )
                continue
            attempt["state"] = "completed"
            attempt["error"] = None
            attempt["recovery"] = (
                "healthy terminal Evidence recovered after runner interruption"
            )
        for attempt in running:
            if (
                attempt.get("phase") == "role_profile"
                and attempt.get("state") in {"failed", "invalid"}
            ):
                attempt["failure_classification"] = (
                    _classify_role_profile_failure(attempt)
                )
        self._prior_run_seconds = sum(
            float(attempt.get("duration_seconds") or 0)
            for attempt in self._state["attempts"]
        )
        self._save()

    def _record_post_run_quiescence(self, attempt: dict[str, Any]) -> None:
        quiescent, active, quiescence_seconds = _wait_for_runtime_quiescence(
            timeout_seconds=self.run_policy.post_run_quiescence_timeout_seconds
        )
        attempt["post_run_quiescence_seconds"] = quiescence_seconds
        attempt["post_run_quiescence_state"] = (
            "clean" if quiescent else "timed_out"
        )
        if attempt["duration_seconds"] is not None:
            attempt["duration_seconds"] += quiescence_seconds
        if quiescent:
            return
        detail = (
            "launched runtime did not release Ray/GPU resources within "
            f"{self.run_policy.post_run_quiescence_timeout_seconds:.1f}s: {active}"
        )
        if attempt["state"] == "completed":
            attempt["state"] = "invalid"
        attempt["error"] = (
            f"{attempt['error']}; {detail}" if attempt["error"] else detail
        )

    def _run_candidate(
        self,
        candidate: CompiledTopologyCandidate,
        *,
        phase: str,
        calibration_workload: bool = True,
    ) -> dict[str, Any] | None:
        remaining = self._remaining_budget()
        if remaining <= 0:
            self._state["stop_reason"] = "total_gpu_time_budget_exhausted"
            self._save()
            return None
        active = _active_runtime_processes()
        if active["ray"] or active["gpu"]:
            raise RuntimeError(
                "refusing to start a calibration attempt while another Ray/GPU runtime exists: "
                f"{active}"
            )
        repeat = len(self._attempts_for(candidate.candidate_id)) + 1
        run_dir = self.output_root / "runs" / candidate.label / f"attempt-{repeat:02d}"
        if run_dir.exists():
            raise RuntimeError(f"refusing to overwrite calibration attempt: {run_dir}")
        run_dir.mkdir(parents=True)
        command = [
            sys.executable,
            "-m",
            "aspec.train_entry",
            "--config",
            str(self.config_path),
            *(self.base_overrides if calibration_workload else ()),
            *candidate.overrides,
            f"output_dir={run_dir}",
        ]
        expected_executed_candidate_id = (
            candidate.candidate_id
            if calibration_workload
            else self.formal_candidate_ids[candidate.candidate_id]
        )
        attempt = {
            "candidate_id": candidate.candidate_id,
            "phase": phase,
            "repeat": repeat,
            "run_dir": str(run_dir),
            "command": command,
            "state": "running",
            "started_at_unix": time.time(),
            "returncode": None,
            "duration_seconds": None,
            "error": None,
            "calibration_workload": calibration_workload,
            "expected_executed_candidate_id": expected_executed_candidate_id,
            "environment_overrides": {
                "ASPEC_HARDWARE_LABEL": self.hardware_cohort["gpu_model"]
            },
        }
        self._state["attempts"].append(attempt)
        self._save()
        log_path = run_dir / "train.log"
        started = time.monotonic()
        timeout = min(self.run_policy.per_run_timeout_seconds, max(1.0, remaining))
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.config_path.parents[1]),
                env={
                    **os.environ,
                    "ASPEC_HARDWARE_LABEL": self.hardware_cohort["gpu_model"],
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
                returncode = process.returncode
                attempt["error"] = f"run exceeded {timeout:.1f}s timeout"
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
                attempt["returncode"] = process.returncode
                attempt["duration_seconds"] = time.monotonic() - started
                attempt["state"] = "interrupted"
                attempt["error"] = "automatic topology runner was interrupted"
                self._record_post_run_quiescence(attempt)
                self._save()
                raise
        attempt["returncode"] = returncode
        attempt["duration_seconds"] = time.monotonic() - started
        evidence_path = run_dir / "evidence.json"
        attempt["evidence_path"] = str(evidence_path)
        if returncode == 0 and evidence_path.is_file():
            try:
                measurement = load_candidate_measurement(evidence_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                attempt["state"] = "invalid"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            else:
                attempt["executed_candidate_id"] = measurement.candidate_id
                if measurement.candidate_id != expected_executed_candidate_id:
                    attempt["state"] = "invalid"
                    attempt["error"] = (
                        "executed Plan does not match the session's exact expected "
                        f"semantics: expected={expected_executed_candidate_id}, "
                        f"observed={measurement.candidate_id}"
                    )
                elif not measurement.valid:
                    attempt["state"] = "invalid"
                    attempt["error"] = "Evidence did not pass calibration health gates"
                else:
                    attempt["state"] = "completed"
        else:
            attempt["state"] = "failed"
            if attempt["error"] is None:
                attempt["error"] = f"training subprocess exited with {returncode}"

        self._record_post_run_quiescence(attempt)
        if phase == "role_profile" and attempt["state"] in {"failed", "invalid"}:
            attempt["failure_classification"] = _classify_role_profile_failure(
                attempt
            )
        self._save()
        return attempt

    def _completed_measurements(
        self,
        candidate_ids: set[str] | None = None,
        *,
        phases: set[str] | None = None,
    ) -> list[CandidateMeasurement]:
        measurements = []
        for attempt in self._state["attempts"]:
            if attempt["state"] != "completed":
                continue
            if candidate_ids is not None and attempt["candidate_id"] not in candidate_ids:
                continue
            if phases is not None and attempt["phase"] not in phases:
                continue
            measurements.append(load_candidate_measurement(attempt["evidence_path"]))
        return measurements

    def _select_finalist(self, finalist_ids: set[str]) -> dict[str, Any]:
        selection_phases = {"role_profile", "short_e2e"}
        attempts_by_candidate = {
            candidate_id: self._phase_attempts(candidate_id, selection_phases)
            for candidate_id in finalist_ids
        }
        failed = {
            candidate_id
            for candidate_id, attempts in attempts_by_candidate.items()
            if any(attempt["state"] in {"failed", "invalid"} for attempt in attempts)
        }
        eligible = {
            candidate_id
            for candidate_id, attempts in attempts_by_candidate.items()
            if candidate_id not in failed
            and any(attempt["state"] == "completed" for attempt in attempts)
        }
        measurements = self._completed_measurements(
            eligible, phases=selection_phases
        )
        if not measurements:
            raise ValueError("no stable finalist has valid end-to-end Evidence")
        environments = {row.environment_id for row in measurements}
        hardware_cohorts = {row.hardware_cohort_id for row in measurements}
        workloads = {row.workload_id for row in measurements}
        if (
            len(environments) != 1
            or len(hardware_cohorts) != 1
            or len(workloads) != 1
        ):
            raise ValueError(
                "finalist Evidence is not environment/hardware/workload comparable"
            )
        selection = select_candidate(
            measurements,
            equivalence_fraction=self.run_policy.equivalence_fraction,
        )
        selection["environment_id"] = next(iter(environments))
        selection["hardware_cohort_id"] = next(iter(hardware_cohorts))
        selection["workload_id"] = next(iter(workloads))
        selection["ineligible_candidates_with_failed_attempts"] = sorted(failed)
        selection["eligible_candidate_ids"] = sorted(eligible)
        return selection

    def _profile_candidate_ids(
        self, bundle_id: str, target_tp: int
    ) -> tuple[str, ...]:
        return tuple(
            candidate_id
            for candidate_id in self.study.profile_candidate_ids
            if (
                (candidate := self.study.candidate(candidate_id)).bundle_id
                == bundle_id
                and candidate.target_tensor_parallel_size == target_tp
            )
        )

    def _load_profile_tier(
        self, candidate_ids: Sequence[str]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        observations: list[Any] = []
        failures: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            attempts = self._phase_attempts(candidate_id, {"role_profile"})
            completed = [
                attempt for attempt in attempts if attempt["state"] == "completed"
            ]
            if completed:
                try:
                    observation = load_service_observation(
                        self.study, completed[-1]["evidence_path"]
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    failures.append(
                        {
                            "candidate_id": candidate_id,
                            "states": ["invalid_service_profile"],
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "failure_classifications": [
                                {
                                    "kind": "unclassified_failure",
                                    "signals": ["invalid_service_profile"],
                                    "source": "evidence",
                                }
                            ],
                        }
                    )
                else:
                    observations.append(observation)
                continue

            classifications = []
            for attempt in attempts:
                if attempt["state"] not in {"failed", "invalid"}:
                    continue
                classification = attempt.get("failure_classification")
                if classification is None:
                    classification = _classify_role_profile_failure(attempt)
                    attempt["failure_classification"] = classification
                classifications.append(classification)
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "states": [attempt["state"] for attempt in attempts],
                    "errors": [
                        attempt.get("error")
                        for attempt in attempts
                        if attempt.get("error")
                    ],
                    "failure_classifications": classifications,
                }
            )
        return observations, failures

    @staticmethod
    def _tier_allows_tp_fallback(
        candidate_ids: Sequence[str], failures: Sequence[dict[str, Any]]
    ) -> bool:
        if not candidate_ids or len(failures) != len(candidate_ids):
            return False
        return all(
            failure["failure_classifications"]
            and all(
                classification.get("kind") == "target_capacity_infeasible"
                for classification in failure["failure_classifications"]
            )
            for failure in failures
        )

    def _resolve_target_tp_profiles(
        self,
    ) -> tuple[list[Any], list[dict[str, Any]]] | None:
        """Select the first viable TP tier in every expert capability bundle."""

        all_observations: list[Any] = []
        all_failures: list[dict[str, Any]] = []
        resolution: list[dict[str, Any]] = []
        for bundle in self.study.bundles:
            bundle_result: dict[str, Any] = {
                "bundle_id": bundle.bundle_id,
                "bundle_name": bundle.name,
                "tp_ladder": list(bundle.target_tensor_parallel_sizes),
                "selected_target_tensor_parallel_size": None,
                "state": "unresolved",
                "tiers": [],
            }
            resolution.append(bundle_result)
            selected = False
            for target_tp in bundle.target_tensor_parallel_sizes:
                candidate_ids = self._profile_candidate_ids(
                    bundle.bundle_id, target_tp
                )
                tier_result: dict[str, Any] = {
                    "target_tensor_parallel_size": target_tp,
                    "profile_candidate_ids": list(candidate_ids),
                    "state": "pending",
                }
                bundle_result["tiers"].append(tier_result)
                if not candidate_ids:
                    tier_result["state"] = "no_legal_profile_candidates"
                    continue
                for candidate_id in candidate_ids:
                    if self._should_run_phase(candidate_id, "role_profile"):
                        if (
                            self._run_candidate(
                                self.study.candidate(candidate_id),
                                phase="role_profile",
                            )
                            is None
                        ):
                            tier_result["state"] = "budget_stopped"
                            self._state["target_tp_resolution"] = resolution
                            self._save()
                            return None
                observations, failures = self._load_profile_tier(candidate_ids)
                tier_result["healthy_profile_candidate_ids"] = [
                    row.candidate_id for row in observations
                ]
                tier_result["failures"] = failures
                all_failures.extend(failures)
                if observations:
                    tier_result["state"] = "selected"
                    bundle_result["selected_target_tensor_parallel_size"] = target_tp
                    bundle_result["state"] = "selected"
                    all_observations.extend(observations)
                    selected = True
                    break
                if self._tier_allows_tp_fallback(candidate_ids, failures):
                    tier_result["state"] = "target_capacity_fallback"
                    continue
                tier_result["state"] = "failed_without_target_capacity_evidence"
                self._state["target_tp_resolution"] = resolution
                self._save()
                raise ValueError(
                    "role-profile tier failed without explicit Target-capacity "
                    f"evidence; refusing to increase TP: bundle={bundle.name}, "
                    f"target_tp={target_tp}"
                )
            if selected:
                selected_index = len(bundle_result["tiers"])
                for target_tp in bundle.target_tensor_parallel_sizes[selected_index:]:
                    bundle_result["tiers"].append(
                        {
                            "target_tensor_parallel_size": target_tp,
                            "profile_candidate_ids": list(
                                self._profile_candidate_ids(bundle.bundle_id, target_tp)
                            ),
                            "state": "skipped_after_lower_tp_selected",
                        }
                    )
            else:
                bundle_result["state"] = "exhausted_without_viable_target_tp"
        self._state["target_tp_resolution"] = resolution
        self._save()
        return all_observations, all_failures

    def run(self) -> dict[str, Any]:
        """Advance or resume the bounded automatic calibration session."""

        if self._state.get("state") in {"completed", "failed"}:
            return self._state
        try:
            return self._run_session()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._state["state"] = "failed"
            self._state["finished_at_unix"] = time.time()
            self._state["stop_reason"] = "automatic_topology_session_failed"
            self._state["error"] = f"{type(exc).__name__}: {exc}"
            self._save()
            raise

    def _run_session(self) -> dict[str, Any]:
        self._recover_orphaned_attempts()
        resolved_profiles = self._resolve_target_tp_profiles()
        if resolved_profiles is None:
            return self._state
        observations, failed_profiles = resolved_profiles
        if not observations:
            raise ValueError("no role-profile family produced valid service evidence")
        ranking = rank_topology_candidates(self.study, observations)
        ranking["failed_role_profiles"] = failed_profiles
        ranking["target_tp_resolution"] = self._state["target_tp_resolution"]
        self._state["ranking"] = ranking
        self._save()

        finalist_ids = set(ranking["finalist_candidate_ids"])
        for candidate_id in ranking["finalist_candidate_ids"]:
            if self._should_run_selection_candidate(candidate_id):
                if (
                    self._run_candidate(
                        self.study.candidate(candidate_id), phase="short_e2e"
                    )
                    is None
                ):
                    return self._state
        selection = self._select_finalist(finalist_ids)
        selection["short_window_recommended"] = dict(selection["recommended"])
        equivalent_ids = [
            row["candidate_id"] for row in selection["ranked_equivalent"]
        ]
        confirmations: list[dict[str, Any]] = []
        if self.run_policy.confirm_repeats:
            for candidate_id in equivalent_ids:
                attempts = self._phase_attempts(
                    candidate_id, {"winner_confirmation"}
                )
                if any(
                    attempt["state"] in {"failed", "invalid"}
                    for attempt in attempts
                ):
                    continue
                completed_count = sum(
                    attempt["state"] == "completed" for attempt in attempts
                )
                for _ in range(
                    max(0, self.run_policy.confirm_repeats - completed_count)
                ):
                    attempt = self._run_candidate(
                        self.study.candidate(candidate_id),
                        phase="winner_confirmation",
                        calibration_workload=False,
                    )
                    if attempt is None:
                        return self._state
                    if attempt["state"] != "completed":
                        break

            formal_measurements: list[CandidateMeasurement] = []
            failed_confirmations = []
            for candidate_id in equivalent_ids:
                attempts = self._phase_attempts(
                    candidate_id, {"winner_confirmation"}
                )
                failed = any(
                    attempt["state"] in {"failed", "invalid"}
                    for attempt in attempts
                )
                completed = [
                    attempt
                    for attempt in attempts
                    if attempt["state"] == "completed"
                ]
                if failed or len(completed) < self.run_policy.confirm_repeats:
                    failed_confirmations.append(candidate_id)
                    continue
                for attempt in completed:
                    observed = load_candidate_measurement(attempt["evidence_path"])
                    if observed.environment_id != selection["environment_id"]:
                        raise ValueError(
                            "formal-window confirmation environment differs from "
                            "short-window Evidence"
                        )
                    if (
                        observed.hardware_cohort_id
                        != selection["hardware_cohort_id"]
                    ):
                        raise ValueError(
                            "formal-window confirmation hardware cohort differs from "
                            "short-window Evidence"
                        )
                    comparable = replace(observed, candidate_id=candidate_id)
                    formal_measurements.append(comparable)
                    payload = comparable.as_dict()
                    payload["executed_candidate_id"] = observed.candidate_id
                    confirmations.append(payload)
            if not formal_measurements:
                raise ValueError(
                    "no equivalent short-window finalist passed formal confirmation"
                )
            formal_workloads = {
                measurement.workload_id for measurement in formal_measurements
            }
            if len(formal_workloads) != 1:
                raise ValueError(
                    "formal-window confirmations do not share one Workload"
                )
            formal_selection = select_candidate(
                formal_measurements,
                equivalence_fraction=self.run_policy.equivalence_fraction,
            )
            selection["formal_window_selection"] = formal_selection
            selection["failed_formal_confirmation_candidate_ids"] = sorted(
                failed_confirmations
            )
            selection["recommended"] = formal_selection["recommended"]
        selection["formal_window_confirmations"] = confirmations
        self._state["selection"] = selection
        self._state["state"] = "completed"
        self._state["finished_at_unix"] = time.time()
        self._state["stop_reason"] = (
            "winner_confirmed_with_formal_end_to_end_evidence"
            if confirmations
            else "winner_selected_from_short_end_to_end_evidence"
        )
        self._save()
        return self._state
