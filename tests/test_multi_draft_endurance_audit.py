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

import json

import pytest

from tools.validation import multi_draft_endurance_summary as audit


def _session(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "build_run_report", lambda _: {"health": "healthy"})
    evidence = {
        "state": "completed_with_failed_experiments",
        "cleanup_errors": [],
        "read_owners_fenced": True,
        "wall_seconds": 3600,
        "shared_source": {
            "outstanding_observations": 0,
            "outstanding_payload_bytes": 0,
            "pending_production": 0,
            "retiring_experiments": [],
            "active_experiments": [],
        },
        "inference_manager": {"inflight_samples": 0},
        "experiments": {
            "draft_b": {
                "state": "failed",
                "error": "injected isolated failure after slow-consumer recovery",
            }
        },
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    for name, step in (("draft_a", 32), ("draft_b", 16)):
        base = tmp_path / name / "checkpoints"
        checkpoint = base / f"iter_{step + 1:07d}"
        checkpoint.mkdir(parents=True)
        (base / "latest_checkpointed_iteration.txt").write_text(str(step + 1))
        state = {
            "optimizer_step": step,
            "dataset_epoch": step // 8,
            "committed_work_ids": list(map(str, range(step * 4))),
        }
        (checkpoint / "work_state.json").write_text(json.dumps(state))
    rows = []
    for index, (first, second) in enumerate(((0, 0), (4, 2), (12, 8), (20, 16), (32, 16))):
        children = {
            name: {"lifecycle": {"optimizer_steps_committed": step}, "tracked_epochs": 8}
            for name, step in (("draft_a", first), ("draft_b", second))
        }
        rows.append(
            {
                "elapsed_seconds": index * 800,
                "source": {"experiments": children, "outstanding_observations": 12},
                "processes": [{"pid": 12, "name": "ray::ObservedSourceState", "rss_bytes": 1000}],
            }
        )
    (tmp_path / "endurance-observations.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return tmp_path


def test_endurance_audit_checks_native_lifecycle_and_shared_committed_prefix(tmp_path, monkeypatch):
    root = _session(tmp_path, monkeypatch)
    result = audit.summarize(root, slow_steps=8, fail_step=16)
    assert result["checkpoint_progress"]["draft_a"]["committed_work"] == 128
    assert result["checkpoint_progress"]["draft_b"]["committed_work"] == 64
    assert result["maximum_step_lead_during_slow_phase"] == 2
    assert result["shared_committed_prefix_preserved"]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("read_owners_fenced", False, "fence"),
        ("cleanup_errors", ["unsafe"], "fence"),
        ("state", "failed", "isolate"),
    ],
)
def test_endurance_audit_cannot_upgrade_unsafe_or_shared_failure(
    tmp_path, monkeypatch, field, value, match
):
    root = _session(tmp_path, monkeypatch)
    path = root / "evidence.json"
    evidence = json.loads(path.read_text())
    evidence[field] = value
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match=match):
        audit.summarize(root, slow_steps=8, fail_step=16)


def test_endurance_audit_rejects_an_unrelated_draft_failure(tmp_path, monkeypatch):
    root = _session(tmp_path, monkeypatch)
    path = root / "evidence.json"
    evidence = json.loads(path.read_text())
    evidence["experiments"]["draft_b"]["error"] = "unexpected OOM"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="intended injection"):
        audit.summarize(root, slow_steps=8, fail_step=16)
