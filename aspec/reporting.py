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

"""Read-only summaries for completed or failed ASpec run evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from aspec.evidence import EVIDENCE_SCHEMA_VERSION
from aspec.planning import PLAN_SCHEMA_VERSION

RUN_REPORT_SCHEMA_VERSION = "aspec.run-report/v0alpha1"


class RunReportError(ValueError):
    """Raised when run artifacts cannot support a trustworthy report."""


def _is_safe_experiment_name(name: object) -> bool:
    """A printable path component, not an ASCII-only hyperparameter label."""
    return (
        isinstance(name, str) and bool(name) and name == name.strip()
        and name.isprintable() and "/" not in name and "\\" not in name
        and name not in {".", "..", "config.yaml", "plan.json", "evidence.json"}
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunReportError(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunReportError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunReportError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_evidence_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "evidence.json"
    if not resolved.is_file():
        raise RunReportError(f"run evidence does not exist: {resolved}")
    return resolved


def _validate_plan(path: Path, evidence_plan_id: str, schema=PLAN_SCHEMA_VERSION) -> dict[str, Any]:
    if not path.is_file():
        raise RunReportError(f"plan artifact does not exist: {path}")
    plan = _read_object(path, "run plan")
    if plan.get("schema_version") != schema:
        raise RunReportError(
            f"unsupported plan schema {plan.get('schema_version')!r}; "
            f"expected {schema!r}"
        )
    declared_plan_id = plan.get("plan_id")
    payload = dict(plan)
    payload.pop("plan_id", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    computed_plan_id = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if declared_plan_id != computed_plan_id:
        raise RunReportError(
            "plan content does not match its declared identity: "
            f"{declared_plan_id!r} != {computed_plan_id!r}"
        )
    if declared_plan_id != evidence_plan_id:
        raise RunReportError(
            "plan and evidence identities do not match: "
            f"{declared_plan_id!r} != {evidence_plan_id!r}"
        )
    return plan


def build_run_report(path: str | Path) -> dict[str, Any]:
    """Validate one run directory and return a compact canonical summary."""

    evidence_path = _resolve_evidence_path(path)
    evidence = _read_object(evidence_path, "run evidence")
    if evidence.get("schema_version") == "aspec.multi-draft-evidence/v1":
        return _build_multi_draft_report(evidence_path, evidence)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise RunReportError(
            f"unsupported evidence schema {evidence.get('schema_version')!r}; "
            f"expected {EVIDENCE_SCHEMA_VERSION!r}"
        )

    plan_id = evidence.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.startswith("sha256:"):
        raise RunReportError("run evidence has no valid plan_id")
    plan_path = evidence_path.parent / "plan.json"
    _validate_plan(plan_path, plan_id)

    capabilities = evidence.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, dict) and isinstance(item.get("name"), str)
        for item in capabilities
    ):
        raise RunReportError("run evidence capabilities must be a list of named objects")
    required = [item for item in capabilities if item.get("requirement") == "required"]
    active = [item["name"] for item in required if item.get("effective_state") == "active"]
    pending = [item["name"] for item in required if item.get("effective_state") != "active"]

    state = evidence.get("state")
    health = evidence.get("health")
    if state not in {"running", "completed", "failed", "cancelled"}:
        raise RunReportError(f"run evidence has unsupported state: {state!r}")
    if health not in {"pending", "healthy", "incomplete_evidence", "failed"}:
        raise RunReportError(f"run evidence has unsupported health: {health!r}")
    if health == "healthy" and (state != "completed" or pending):
        raise RunReportError(
            "healthy evidence must be completed with every required capability active"
        )
    declared_pending = evidence.get("required_capabilities_pending")
    if declared_pending is not None:
        if not isinstance(declared_pending, list) or not all(
            isinstance(item, str) for item in declared_pending
        ):
            raise RunReportError("required_capabilities_pending must be a list of names")
        if sorted(declared_pending) != sorted(pending):
            raise RunReportError(
                "required_capabilities_pending disagrees with capability evidence"
            )

    observations = evidence.get("observations")
    if not isinstance(observations, Mapping):
        observations = {}
    training = observations.get("training")
    if not isinstance(training, Mapping):
        training = {}
    cleanup = observations.get("cleanup")
    if not isinstance(cleanup, Mapping):
        cleanup = {}
    lifecycle = cleanup.get("work_lifecycle")
    if not isinstance(lifecycle, Mapping):
        lifecycle = {}
    final_metrics = training.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        final_metrics = {}
    performance = training.get("performance")
    if not isinstance(performance, Mapping):
        performance = {}

    fallbacks = evidence.get("fallbacks")
    omissions = evidence.get("omissions")
    if not isinstance(fallbacks, list) or not isinstance(omissions, list):
        raise RunReportError("fallbacks and omissions must be lists")

    source = evidence.get("source")
    if not isinstance(source, Mapping):
        source = {}
    tracking_path = evidence_path.parent / "tracking.json"
    tracking = _read_object(tracking_path, "experiment tracking") if tracking_path.is_file() else None
    if tracking is not None and (
        tracking.get("schema_version") != "aspec.experiment-tracking/v1"
        or tracking.get("run_id") != evidence.get("run_id")
        or tracking.get("experiment") != evidence_path.parent.name
        or not isinstance(tracking.get("tensorboard"), dict)
        or not isinstance(tracking.get("wandb"), dict)
        or tracking.get("state") not in {"initializing", "active", "closed", "failed"}
    ):
        raise RunReportError("experiment tracking disagrees with run identity")
    return {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run_id": evidence.get("run_id"),
        "plan_id": plan_id,
        "state": state,
        "health": health,
        "tracking": tracking,
        "started_at": evidence.get("started_at"),
        "finished_at": evidence.get("finished_at"),
        "duration_seconds": evidence.get("duration_seconds"),
        "required_capabilities": {
            "active": active,
            "pending": pending,
            "active_count": len(active),
            "total_count": len(required),
        },
        "fallbacks": fallbacks,
        "omissions": [
            item.get("capability", item) if isinstance(item, Mapping) else item
            for item in omissions
        ],
        "training": {
            "start_step": training.get("start_step"),
            "completed_step": training.get("completed_step"),
            "no_new_work": training.get("no_new_work", False),
            "optimizer_steps_observed": training.get("optimizer_steps_observed"),
            "requested_source_work_budget": training.get(
                "requested_source_work_budget"
            ),
            "source_work_budget": training.get("source_work_budget"),
            "effective_source_work": training.get("effective_source_work"),
            "dropped_source_work": training.get("dropped_source_work"),
            "planned_dropped_source_work": training.get(
                "planned_dropped_source_work"
            ),
            "source_work_alignment": training.get("source_work_alignment"),
            "source_work_tail_policy": training.get("source_work_tail_policy"),
            "final_metrics": dict(final_metrics),
            "performance": dict(performance),
            "evaluation": training.get("evaluation"),
        },
        "cleanup": {
            "state": cleanup.get("state"),
            "work_lifecycle": {
                key: lifecycle.get(key)
                for key in (
                    "state",
                    "produced_outputs",
                    "committed_work",
                    "discarded_outputs",
                    "resident_outputs",
                    "resident_bytes",
                )
                if key in lifecycle
            },
        },
        "error": evidence.get("error"),
        "source": {
            "aspec": source.get("aspec"),
            "packages": source.get("packages"),
            "backends": source.get("backends"),
            "host": source.get("host"),
        },
        "artifacts": {
            "plan": str(plan_path),
            "evidence": str(evidence_path),
            **({"tracking": str(tracking_path)} if tracking is not None else {}),
        },
    }


def _build_multi_draft_report(evidence_path, evidence):
    plan = _validate_plan(evidence_path.parent / "plan.json", evidence.get("plan_id"),
                          "aspec.multi-draft-plan/v1")
    if evidence.get("state") not in {"running", "completed", "failed", "completed_with_failed_experiments"}:
        raise RunReportError("unsupported multi-Draft session state")
    experiments = {}
    for name, child_plan in plan["experiments"].items():
        if not _is_safe_experiment_name(name):
            raise RunReportError("unsafe experiment identity in session plan")
        child_path = evidence_path.parent / name / "evidence.json"
        if not child_path.exists() and evidence["state"] != "completed":
            experiments[name] = {"state": "uninitialized", "health": "pending"}
            continue
        child = build_run_report(child_path)
        if child["plan_id"] != child_plan["plan_id"] or child["run_id"] != evidence["run_id"]:
            raise RunReportError("experiment evidence disagrees with session identity")
        experiments[name] = child
    healthy = (evidence["state"] == "completed" and experiments
               and all(child["health"] == "healthy" for child in experiments.values())
               and not evidence.get("cleanup_errors")
               and evidence.get("read_owners_fenced") is not False)
    return {
        "schema_version": "aspec.multi-draft-report/v1",
        "run_id": evidence["run_id"], "plan_id": plan["plan_id"],
        "state": evidence["state"], "health": "healthy" if healthy else "incomplete_or_failed",
        "wall_seconds": evidence.get("wall_seconds"),
        "budget_gpu_hours": evidence.get("budget_gpu_hours", evidence.get("reserved_gpu_hours")),
        "shared_source": evidence.get("shared_source"),
        "read_owners_fenced": evidence.get("read_owners_fenced"),
        "experiments": experiments, "error": evidence.get("error"),
        "cleanup_errors": evidence.get("cleanup_errors", []),
    }


def render_run_report(report: Mapping[str, Any]) -> str:
    """Render the compact report for a terminal without hiding missing evidence."""

    if report["schema_version"] == "aspec.multi-draft-report/v1":
        lines = [f"ASpec multi-Draft session: {report['run_id']}",
                 f"state: {report['state']}  health: {report['health']}",
                 f"wall_seconds: {report['wall_seconds']}",
                 f"budget_gpu_hours: {report['budget_gpu_hours']}",
                 f"shared_source: {report['shared_source']}"]
        for name, child in report["experiments"].items():
            lines.append(f"experiment {name}: {child['state']} / {child['health']}")
            if "training" in child:
                lines.append(render_run_report(child))
        return "\n".join(lines)
    required = report["required_capabilities"]
    training = report["training"]
    cleanup = report["cleanup"]
    lifecycle = cleanup["work_lifecycle"]
    performance = training["performance"]
    final_metrics = training["final_metrics"]
    lines = [
        f"ASpec run: {report.get('run_id') or '-'}",
        f"state: {report.get('state')}  health: {report.get('health')}",
        f"plan: {report.get('plan_id')}",
        f"duration_seconds: {report.get('duration_seconds')}",
        (
            "required_capabilities: "
            f"{required['active_count']}/{required['total_count']} active"
        ),
        f"fallbacks: {len(report['fallbacks'])}",
        "omissions: " + (", ".join(map(str, report["omissions"])) or "none"),
        f"optimizer_steps_observed: {training.get('optimizer_steps_observed')}",
    ]
    if required["pending"]:
        lines.append("required_pending: " + ", ".join(required["pending"]))
    if training.get("start_step") is not None and training.get("completed_step") is not None:
        lines.append(f"optimizer_progress: {training['start_step']} -> {training['completed_step']}")
    if training.get("requested_source_work_budget") is not None:
        lines.append(
            "source_work: "
            f"requested={training['requested_source_work_budget']} "
            f"scheduled={training['source_work_budget']} "
            f"effective={training['effective_source_work']} "
            f"dropped={training['dropped_source_work']} "
            f"alignment={training['source_work_alignment']} "
            f"policy={training['source_work_tail_policy']}"
        )
    if training.get("no_new_work"):
        lines.append("training: checkpoint horizon already complete; no new work")
    tracking = report.get("tracking")
    if tracking is not None:
        lines.append(f"tracking: {tracking['state']}; tensorboard={tracking['tensorboard'].get('log_dir')}; "
                     f"wandb_run={tracking['wandb'].get('run_id')}")
    evaluation = training.get("evaluation")
    if evaluation and evaluation.get("enabled"):
        lines.append(f"held_out_eval: {len(evaluation.get('history', []))} points; proxy_only=true")
        best = evaluation.get("best_checkpoint")
        lines.append(f"best_checkpoint: {best.get('checkpoint') if best else None}")
    if performance:
        lines.append(
            "performance: "
            f"state={performance.get('state')} "
            f"committed_work_per_second={performance.get('committed_work_per_second')}"
        )
    summary_metrics = {
        key: final_metrics[key]
        for key in ("train/avg_loss", "train/grad_norm", "train/avg_acc")
        if key in final_metrics
    }
    if summary_metrics:
        metrics = " ".join(
            f"{key}={value}" for key, value in summary_metrics.items()
        )
        lines.append(f"final_metrics: {metrics}")
    lines.append(f"cleanup: {cleanup.get('state')}")
    if lifecycle:
        lines.append(
            "lifecycle: "
            f"state={lifecycle.get('state')} "
            f"produced={lifecycle.get('produced_outputs')} "
            f"committed={lifecycle.get('committed_work')} "
            f"discarded={lifecycle.get('discarded_outputs')} "
            f"resident_outputs={lifecycle.get('resident_outputs')} "
            f"resident_bytes={lifecycle.get('resident_bytes')}"
        )
    if report.get("error"):
        error = report["error"]
        lines.append(f"error: {error.get('type')}: {error.get('message')}")
        for item in error.get("backend_logs", []):
            lines.append(f"backend_log: {item['path']}")
            lines.append(item.get("tail", item.get("read_error", "")).rstrip())
    return "\n".join(lines)
