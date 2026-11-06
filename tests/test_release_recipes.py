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

import copy
import json

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan
from tools import recipes


@pytest.mark.parametrize("name", recipes.catalog())
def test_catalog_plans(name, monkeypatch):
    monkeypatch.chdir(recipes.ROOT)
    monkeypatch.setenv(
        "ASPEC_TARGET_MODEL",
        "Qwen/Qwen3.8-27B" if name == "qwen27b-long" else recipes.catalog()[name]["model"],
    )
    if name.startswith("ling"):
        monkeypatch.setenv(
            "ASPEC_TARGET_MODEL", "inclusionAI/" + recipes.catalog()[name]["model"]
        )
    monkeypatch.setenv("ASPEC_NODE_0_IP", "192.0.2.1")
    monkeypatch.setenv("ASPEC_NODE_1_IP", "192.0.2.2")
    monkeypatch.setenv("ASPEC_SHARED_ROOT", "/tmp/aspec-test")
    if name == "multinode":
        monkeypatch.setenv("ASPEC_SHARED_ROOT", str(recipes.ROOT.parent))
        monkeypatch.setenv("ASPEC_TARGET_MODEL", str(recipes.ROOT.parent / "models/Qwen3-8B"))
        monkeypatch.setenv("ASPEC_TRAIN_DATA", str(recipes.ROOT.parent / "train.parquet"))
        monkeypatch.setenv("ASPEC_EVAL_DATA", str(recipes.ROOT.parent / "eval.parquet"))
    for path in recipes.catalog()[name]["configs"]:
        config = load_config(str(recipes.ROOT / path))
        if name == "shared-target":
            from aspec.multi_draft import compile_multi_draft_plan

            compile_multi_draft_plan(config)
        else:
            compile_execution_plan(config)
        if name != "quickstart":
            assert config.training.num_train_steps is None
            assert config.training.lr_total_steps is None
            assert config.training.num_epochs == 3
            assert config.training.save_final_checkpoint
            assert config.dataset.eval_data_path
        raw = OmegaConf.load(recipes.ROOT / path)
        actual = (
            raw.inference.inference_num_gpus
            + raw.training.training_num_gpus_per_node
            * raw.training.training_num_nodes
            * (len(raw.get("experiments", [])) or 1)
        )
        assert actual == recipes.catalog()[name]["gpus"]


@pytest.mark.parametrize("name", ["dspark", "shared-target"])
def test_qwen8b_recipes_plan_without_target_override(name, monkeypatch):
    monkeypatch.chdir(recipes.ROOT)
    monkeypatch.delenv("ASPEC_TARGET_MODEL", raising=False)
    for path in recipes.catalog()[name]["configs"]:
        config = load_config(str(recipes.ROOT / path))
        assert config.model.target_model_path == "Qwen/Qwen3-8B"
        if name == "shared-target":
            from aspec.multi_draft import compile_multi_draft_plan

            compile_multi_draft_plan(config)
        else:
            compile_execution_plan(config)


def test_catalog_exactly_covers_formal_files():
    entries = recipes.catalog()
    assert len(entries) == 12
    declared = {p for entry in entries.values() for p in entry["configs"]}
    assert declared == {
        str(p.relative_to(recipes.ROOT)) for p in (recipes.ROOT / "configs/recipes").glob("*.yaml")
    }


def test_prepare_validates_and_never_overwrites(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )
    destination = tmp_path / "out.jsonl"
    recipes.prepare(source, destination)
    assert source.read_bytes() == destination.read_bytes()
    with pytest.raises(ValueError, match="existing"):
        recipes.prepare(source, destination)
    source.write_text('{"conversations": []}\n')
    with pytest.raises(ValueError, match="assistant"):
        recipes.prepare(source, tmp_path / "bad.jsonl")


def test_missing_or_identical_data_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ASPEC_EVAL_DATA")
    with pytest.raises(ValueError):
        recipes.check_inputs("dspark")
    source = tmp_path / "data.jsonl"
    source.write_text("{}\n")
    for key in ("ASPEC_TRAIN_DATA", "ASPEC_EVAL_DATA"):
        monkeypatch.setenv(key, str(source))
    with pytest.raises(ValueError, match="differ"):
        recipes.check_inputs("dspark")


@pytest.mark.parametrize("action,tokens", [("train", 128), ("accept", 128), ("accept", 261888)])
def test_long_input_minimum_applies_only_to_acceptance(tmp_path, monkeypatch, action, tokens):
    import sys
    from unittest.mock import Mock

    import pyarrow as pa
    import pyarrow.parquet as pq

    for key, length in (("ASPEC_TRAIN_DATA", tokens), ("ASPEC_EVAL_DATA", 64)):
        path = tmp_path / f"{key}.parquet"
        pq.write_table(pa.table({"input_ids": [[1] * length]}), path)
        monkeypatch.setenv(key, str(path))
    launch = Mock()
    monkeypatch.setattr(recipes, "execute", launch)
    monkeypatch.setattr(recipes, "run_acceptance", launch)
    monkeypatch.setattr(sys, "argv", ["recipes", action, "qwen27b-long", "--output", str(tmp_path / "run")])
    if action == "accept" and tokens < 261888:
        with pytest.raises(ValueError, match="Long-context acceptance"):
            recipes.main()
        launch.assert_not_called()
    else:
        recipes.main()
        launch.assert_called_once()


def healthy():
    return {
        "state": "completed",
        "health": "healthy",
        "fallbacks": [],
        "required_capabilities": {"pending": []},
        "training": {"completed_step": 8, "no_new_work": False, "final_metrics": {"loss": 1.2}},
        "cleanup": {"work_lifecycle": {"resident_outputs": 0, "resident_bytes": 0}},
    }


@pytest.mark.parametrize(
    "kind",
    ["incomplete", "nonfinite", "missing_loss", "no_work", "wrong_step", "ownership", "fallback"],
)
def test_acceptance_rejects_false_passes(kind):
    report = copy.deepcopy(healthy())
    if kind == "incomplete":
        report["health"] = "incomplete_evidence"
    if kind == "nonfinite":
        report["training"]["final_metrics"]["loss"] = float("nan")
    if kind == "missing_loss":
        report["training"]["final_metrics"] = {}
    if kind == "no_work":
        report["training"]["no_new_work"] = True
    if kind == "wrong_step":
        report["training"]["completed_step"] = 4
    if kind == "ownership":
        report["cleanup"]["work_lifecycle"]["resident_outputs"] = 1
    if kind == "fallback":
        report["fallbacks"] = ["host"]
    with pytest.raises(ValueError):
        recipes.assert_report(report, 8)


def test_shared_checks_each_child():
    report = {
        "state": "completed",
        "health": "healthy",
        "experiments": {"a": healthy(), "b": healthy()},
    }
    recipes.assert_report(report, 8)
    report["experiments"]["b"]["training"]["completed_step"] = 4
    with pytest.raises(ValueError):
        recipes.assert_report(report, 8)


def test_fault_check_rejects_startup_failure(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(recipes, "check_inputs", lambda _: {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    def startup_failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["startup"])

    monkeypatch.setattr(recipes, "execute", startup_failure)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Injection was not observed"):
        recipes.fault_check("shared-target", "fail", output)
    assert json.loads((output / "result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("action", ["fail", "slow"])
def test_fault_cli_dispatches_selected_action(action, tmp_path, monkeypatch):
    import sys
    from unittest.mock import Mock

    invoked = Mock()
    monkeypatch.setattr(recipes, "fault_check", invoked)
    output = str(tmp_path / "fault")
    monkeypatch.setattr(
        sys, "argv", ["recipes", "fault-check", "multinode", "--action", action, "--output", output]
    )
    recipes.main()
    invoked.assert_called_once_with("multinode", action, output)


def test_primary_dspark_release_uses_fa4_packing():
    config = load_config(str(recipes.ROOT / recipes.catalog()["dspark"]["configs"][0]))
    assert config.training.attention_backend == "fa4"
    assert config.training.dflash_packing


@pytest.mark.parametrize("threshold", [8, 32])
def test_native_tp4_target_rejects_unreachable_packing_threshold(monkeypatch, threshold):
    from aspec.planning import PlanCompileError

    monkeypatch.setenv("ASPEC_TARGET_MODEL", "inclusionAI/Ling-3.1-flash")
    path = recipes.ROOT / recipes.catalog()["ling-flash31"]["configs"][0]
    config = load_config(str(path), cli_args=[f"training.packing_min_samples_per_dispatch={threshold}"], save_snapshot=False)
    with pytest.raises(PlanCompileError, match="packing supply threshold"):
        compile_execution_plan(config)


@pytest.mark.parametrize("scheduler_matches", [True, False])
@pytest.mark.parametrize("comparison", ["complete", "empty", "missing_master", "zero_coordinates"])
def test_resume_audit_compares_saved_scheduler_state(tmp_path, monkeypatch, scheduler_matches, comparison):
    import aspec.reporting

    stage = tmp_path / "stage"
    first, resumed = stage / "first", stage / "resumed"
    (first / "checkpoints/iter_0000005").mkdir(parents=True)
    (resumed / "checkpoints/iter_0000009").mkdir(parents=True)
    (first / "checkpoints/iter_0000005/work_state.json").write_text(
        json.dumps({"committed_work_ids": ["a"]}))
    (resumed / "checkpoints/iter_0000009/work_state.json").write_text(
        json.dumps({"committed_work_ids": ["a", "b"], "committed_work_count": 2}))
    (first / "recipe-audit-checkpoint-0000004.json").write_text(
        json.dumps({"scheduler": {"last_epoch": 4 if scheduler_matches else 3}}))
    (resumed / "recipe-audit-initial.json").write_text(
        json.dumps({"state": {"model": "start"}, "scheduler": {"last_epoch": 4}}))
    (resumed / "recipe-audit-checkpoint-0000008.json").write_text(
        json.dumps({"state": {"model": "trained"}}))
    state = {key: {"relative_l2": 0, "max_absolute": 0, "coordinates": 1}
             for key in ("model", "master", "exp_avg", "exp_avg_sq")}
    if comparison == "empty":
        state.clear()
    elif comparison == "missing_master":
        state.pop("master")
    elif comparison == "zero_coordinates":
        state["master"]["coordinates"] = 0
    (resumed / "recipe-resume-diff.json").write_text(json.dumps({
        "rng_exact": True, "state": state}))
    monkeypatch.setattr(aspec.reporting, "build_run_report", lambda _: healthy())
    if scheduler_matches and comparison == "complete":
        assert recipes.audit_run(resumed, 8, resumed=True, expected_ranks=1) == [""]
    else:
        message = "LR scheduler" if not scheduler_matches else "Checkpoint state/RNG"
        with pytest.raises(ValueError, match=message):
            recipes.audit_run(resumed, 8, resumed=True, expected_ranks=1)
