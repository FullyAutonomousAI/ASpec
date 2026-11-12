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

import hashlib
import json

import pytest

from aspec.config.train_config import load_config
from aspec.evidence import EVIDENCE_SCHEMA_VERSION
from aspec.planning import compile_execution_plan, save_execution_plan
from aspec.reporting import (
    RUN_REPORT_SCHEMA_VERSION,
    RunReportError,
    build_run_report,
    render_run_report,
)


def _write_run(tmp_path):
    config_path = "configs/examples/sglang_qwen3_8b_dspark.yaml"
    plan = compile_execution_plan(
        load_config(config_path=config_path), config_path=config_path
    )
    output = tmp_path / "run"
    output.mkdir()
    save_execution_plan(plan, output / "plan.json")
    capabilities = []
    for item in plan.to_dict()["capabilities"]:
        item["effective_state"] = (
            "active" if item["requirement"] == "required" else item["effective_state"]
        )
        capabilities.append(item)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": "uuid:test-run",
        "plan_id": plan.plan_id,
        "state": "completed",
        "health": "healthy",
        "started_at": "2026-09-15T00:00:00+00:00",
        "finished_at": "2026-09-15T00:00:10+00:00",
        "duration_seconds": 10.0,
        "source": {"aspec": {"source_tree_sha256": "a" * 64}},
        "capabilities": capabilities,
        "required_capabilities_pending": [],
        "fallbacks": [],
        "omissions": list(plan.to_dict()["omissions"]),
        "observations": {
            "training": {
                "optimizer_steps_observed": 2,
                "requested_source_work_budget": 17,
                "source_work_budget": 16,
                "effective_source_work": 16,
                "dropped_source_work": 1,
                "planned_dropped_source_work": 1,
                "source_work_alignment": 4,
                "source_work_tail_policy": "drop_last_complete_packed_optimizer_step",
                "final_metrics": {"train/avg_loss": 1.25, "train/grad_norm": 2.5},
                "performance": {
                    "state": "measured",
                    "committed_work_per_second": 4.0,
                },
            },
            "cleanup": {
                "state": "completed",
                "work_lifecycle": {
                    "state": "validated",
                    "produced_outputs": 16,
                    "committed_work": 16,
                    "discarded_outputs": 0,
                    "resident_outputs": 0,
                    "resident_bytes": 0,
                },
            },
        },
        "error": None,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    return output, evidence


def test_run_report_validates_artifacts_and_summarizes_claims(tmp_path):
    output, evidence = _write_run(tmp_path)

    report = build_run_report(output)

    assert report["schema_version"] == RUN_REPORT_SCHEMA_VERSION
    assert report["plan_id"] == evidence["plan_id"]
    assert report["required_capabilities"]["pending"] == []
    assert report["required_capabilities"]["active_count"] > 0
    assert report["training"]["performance"]["committed_work_per_second"] == 4.0
    assert report["cleanup"]["work_lifecycle"]["resident_outputs"] == 0
    rendered = render_run_report(report)
    assert "health: healthy" in rendered
    assert "resident_outputs=0" in rendered
    assert (
        "source_work: requested=17 scheduled=16 effective=16 dropped=1 alignment=4"
        in rendered
    )


def test_report_exposes_owned_tracking_identity_and_paths(tmp_path):
    output, evidence = _write_run(tmp_path)
    tracking = {"schema_version": "aspec.experiment-tracking/v1",
                "run_id": evidence["run_id"], "experiment": output.name, "state": "closed",
                "tensorboard": {"enabled": True, "log_dir": str(output / "runs")},
                "wandb": {"enabled": True, "run_id": "independent-run"}}
    (output / "tracking.json").write_text(json.dumps(tracking))
    report = build_run_report(output)
    assert report["tracking"] == tracking
    assert report["artifacts"]["tracking"] == str(output / "tracking.json")
    assert "wandb_run=independent-run" in render_run_report(report)
    tracking["run_id"] = "uuid:other-run"
    (output / "tracking.json").write_text(json.dumps(tracking))
    with pytest.raises(RunReportError, match="tracking disagrees"):
        build_run_report(output)


def test_run_report_includes_held_out_curve_and_selected_asset(tmp_path):
    output, evidence = _write_run(tmp_path)
    evaluation = {"enabled": True, "proxy_only": True,
                  "history": [{"step": 8, "metrics": {"eval/simulated_acc_len": 0.5}}],
                  "best_checkpoint": {"step": 8, "checkpoint": "/run/checkpoints/iter_0000009"}}
    evidence["observations"]["training"]["evaluation"] = evaluation
    (output / "evidence.json").write_text(json.dumps(evidence))
    report = build_run_report(output)
    assert report["training"]["evaluation"] == evaluation
    rendered = render_run_report(report)
    assert "held_out_eval: 1 points; proxy_only=true" in rendered
    assert "best_checkpoint: /run/checkpoints/iter_0000009" in rendered


@pytest.mark.parametrize("name", ["draft_a", "实验 A.lr=0.0006"])
def test_multi_draft_report_keeps_shared_source_separate_and_checks_children(tmp_path, name):
    root = tmp_path / "session"
    root.mkdir()
    output, evidence = _write_run(root)
    output.rename(root / name)
    child_plan = json.loads((root / name / "plan.json").read_text())
    plan = {"schema_version": "aspec.multi-draft-plan/v1", "experiments": {name: child_plan}}
    plan["plan_id"] = "sha256:" + hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    (root / "plan.json").write_text(json.dumps(plan))
    session = {"schema_version": "aspec.multi-draft-evidence/v1", "plan_id": plan["plan_id"],
               "run_id": evidence["run_id"], "state": "completed", "wall_seconds": 10,
               "shared_source": {"published_source_observations": 16}, "cleanup_errors": []}
    (root / "evidence.json").write_text(json.dumps(session))
    report = build_run_report(root)
    assert report["health"] == "healthy"
    assert report["shared_source"]["published_source_observations"] == 16
    assert f"experiment {name}" in render_run_report(report)
    session["read_owners_fenced"] = False
    (root / "evidence.json").write_text(json.dumps(session))
    report = build_run_report(root)
    assert not report["read_owners_fenced"]
    assert report["health"] == "incomplete_or_failed"
    evidence["run_id"] = "uuid:foreign"
    (root / name / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(RunReportError, match="session identity"):
        build_run_report(root)


def test_run_report_rejects_plan_identity_mismatch(tmp_path):
    output, _evidence = _write_run(tmp_path)
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["policy"]["pipeline"]["global_batch_size"] += 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(RunReportError, match="declared identity"):
        build_run_report(output)


def test_run_report_rejects_false_healthy_claim(tmp_path):
    output, evidence = _write_run(tmp_path)
    required = next(
        item for item in evidence["capabilities"] if item["requirement"] == "required"
    )
    required["effective_state"] = "pending_runtime_evidence"
    evidence["required_capabilities_pending"] = [required["name"]]
    (output / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(RunReportError, match="healthy evidence"):
        build_run_report(output)


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_run_report_preserves_unsuccessful_terminal_state(tmp_path, state):
    output, evidence = _write_run(tmp_path)
    evidence["state"] = state
    evidence["health"] = "failed"
    evidence["error"] = {"type": "RuntimeError", "message": "backend stopped"}
    (output / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    report = build_run_report(output)

    assert report["state"] == state
    assert report["health"] == "failed"
    assert "RuntimeError: backend stopped" in render_run_report(report)


def test_report_cli_outputs_json_without_starting_runtime(tmp_path, monkeypatch, capsys):
    output, evidence = _write_run(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["aspec", "--report", str(output), "--report-format", "json"]
    )

    from aspec.train_entry import parse_config

    with pytest.raises(SystemExit) as exc_info:
        parse_config()

    assert exc_info.value.code == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["plan_id"] == evidence["plan_id"]
