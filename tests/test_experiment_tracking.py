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
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aspec.controller import eval as evaluation
from aspec.controller.loop import _write_training_metrics
from aspec.utils import logging as tracking_utils
from aspec.utils import wandb as wandb_utils


def _args(tmp_path, name):
    return SimpleNamespace(
        output_dir=str(tmp_path / name), aspec_run_id="uuid:session1234",
        aspec_experiment_name=name, load_path=None, use_tensorboard=True, use_wandb=True,
        wandb_mode="offline", wandb_key="not-a-real-key", wandb_host=None, wandb_dir=None,
        wandb_group="comparison", wandb_project="tests", wandb_team=None, wandb_random_suffix=True,
    )


def _run(name):
    result = Mock()
    result.id, result.name, result.group = name, name, "comparison_session1"
    result.project, result.dir = "tests", f"logs/{name}"
    result.settings = SimpleNamespace(mode="offline")
    return result


@pytest.fixture
def sinks(monkeypatch):
    writers, runs = [], {}

    def writer(**kwargs):
        result = Mock(log_dir=kwargs["log_dir"])
        writers.append(result)
        return result

    def run(args):
        result = runs[args.aspec_experiment_name] = _run(args.aspec_experiment_name)
        return result

    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=writer))
    monkeypatch.setattr(wandb_utils, "init_wandb_experiment", run)
    return writers, runs


def test_concurrent_drafts_do_not_share_sinks_or_step_axes(tmp_path, monkeypatch, sinks):
    writers, runs = sinks
    trackers = [tracking_utils.ExperimentTracking(_args(tmp_path, name)) for name in ("a", "b")]
    global_log = Mock(side_effect=AssertionError("global W&B used"))
    monkeypatch.setattr("aspec.controller.loop.wandb.log", global_log)
    monkeypatch.setattr("aspec.controller.eval.wandb.log", global_log)
    monkeypatch.setattr("aspec.controller.loop.get_tb_writer", Mock(side_effect=AssertionError("global TB used")))

    def record(tracker, offset):
        with tracker:
            for step in range(offset, offset + 3):
                _write_training_metrics({"train/avg_loss": step / 10, "inference/tokens": 4},
                                        step, step + 2, tracking=tracker)
            group = Mock()
            group.run_eval.return_value = [{"eval/avg_loss": 0.25}]
            evaluation.run_eval(offset + 3, group, True, tracking=tracker)

    with ThreadPoolExecutor(2) as executor:
        list(executor.map(lambda item: record(*item), zip(trackers, (8, 20))))
    for index, (name, offset) in enumerate((("a", 8), ("b", 20))):
        run, writer = runs[name], writers[index]
        assert [call.args[0]["train/step"] for call in run.log.call_args_list[:3]] == list(range(offset, offset + 3))
        assert run.log.call_args_list[-1].args[0]["eval/step"] == offset + 3
        writer.add_scalar.assert_any_call("train/avg_loss", offset / 10, offset)
        writer.add_scalar.assert_any_call("inference/tokens", 4, offset + 2)
        writer.add_scalar.assert_any_call("eval/avg_loss", 0.25, offset + 3)
        run.finish.assert_called_once_with(exit_code=0)
        writer.flush.assert_called_once()
        writer.close.assert_called_once()
        snapshot = json.loads((tmp_path / name / "tracking.json").read_text())
        assert snapshot["last_train_step"] == offset + 2
        assert snapshot["last_eval_step"] == offset + 3
        assert snapshot["metric_calls"] == 4 and snapshot["state"] == "closed"
        trackers[index].close()  # The session's safety cleanup is idempotent.
        run.finish.assert_called_once()
    global_log.assert_not_called()


def test_failure_closes_only_its_own_run_and_retains_original_error(tmp_path, sinks):
    writers, runs = sinks
    a, b = [tracking_utils.ExperimentTracking(_args(tmp_path, name)) for name in ("a", "b")]
    writers[0].flush.side_effect = OSError("flush failed")
    with pytest.raises(RuntimeError, match="training cancelled"):
        with a:
            raise RuntimeError("training cancelled")
    writers[0].close.assert_called_once()
    runs["a"].finish.assert_called_once_with(exit_code=1)
    runs["b"].finish.assert_not_called()
    b.log({"train/avg_loss": 1}, train_step=9)
    b.close()
    runs["b"].finish.assert_called_once_with(exit_code=0)
    assert json.loads(a.path.read_text())["state"] == "failed"


def test_partial_tracking_initialization_releases_writer(tmp_path, monkeypatch, sinks):
    writers, _ = sinks
    monkeypatch.setattr(wandb_utils, "init_wandb_experiment", Mock(side_effect=RuntimeError("SDK unavailable")))
    with pytest.raises(RuntimeError, match="SDK unavailable"):
        tracking_utils.ExperimentTracking(_args(tmp_path, "a"))
    writers[0].flush.assert_called_once()
    writers[0].close.assert_called_once()
    assert json.loads((tmp_path / "a/tracking.json").read_text())["state"] == "failed"


def test_flush_failure_marks_tracking_unsuccessful_but_still_finishes_run(tmp_path, sinks):
    writers, runs = sinks
    tracker = tracking_utils.ExperimentTracking(_args(tmp_path, "a"))
    writers[0].flush.side_effect = OSError("flush failed")
    with pytest.raises(OSError, match="flush failed"):
        tracker.close()
    writers[0].close.assert_called_once()
    runs["a"].finish.assert_called_once_with(exit_code=1)
    snapshot = json.loads(tracker.path.read_text())
    assert snapshot["exit_code"] == 1 and snapshot["state"] == "failed"


def test_wandb_run_handles_share_group_not_identity_or_environment(tmp_path, monkeypatch):
    sdk = Mock()
    sdk.init.side_effect = [_run("a"), _run("b")]
    monkeypatch.setattr(wandb_utils, "wandb", sdk)
    monkeypatch.setenv("WANDB_MODE", "online")
    for name in ("a", "b"):
        args = _args(tmp_path, name)
        args.wandb_dir = str(tmp_path / "custom")
        run = wandb_utils.init_wandb_experiment(args)
        assert args.wandb_run_id == name
        run.define_metric.assert_any_call("train/*", step_metric="train/step")
        run.define_metric.assert_any_call("eval/*", step_metric="eval/step")
    a, b = [call.kwargs for call in sdk.init.call_args_list]
    assert a["group"] == b["group"] == "comparison_session1"
    assert a["name"] == "a" and b["name"] == "b"
    assert a["dir"] != b["dir"]
    assert a["reinit"] == b["reinit"] == "create_new"
    assert a["config"]["wandb_key"] is None
    assert a["sync_tensorboard"] is False and a["mode"] == "offline"
    sdk.login.assert_not_called()
    sdk.define_metric.assert_not_called()
    assert sdk.Settings.call_args.kwargs == {
        "console": "off", "x_disable_stats": True, "api_key": "not-a-real-key",
    }
    online_args = _args(tmp_path, "online")
    online_args.wandb_mode = "online"
    sdk.init.side_effect = None
    sdk.init.return_value = _run("online")
    wandb_utils.init_wandb_experiment(online_args)
    sdk.login.assert_not_called()
    import os

    assert os.environ["WANDB_MODE"] == "online"


def test_disabled_tracking_never_falls_back_to_global_run(tmp_path, monkeypatch):
    args = _args(tmp_path, "a")
    args.use_wandb = args.use_tensorboard = False
    monkeypatch.setattr("aspec.controller.loop.wandb.log", Mock(side_effect=AssertionError("global used")))
    with tracking_utils.ExperimentTracking(args) as tracker:
        _write_training_metrics({"train/avg_loss": 1.0}, 3, 4, tracking=tracker)
    with pytest.raises(RuntimeError, match="after.*closed"):
        tracker.log({}, train_step=5)


def test_wandb_metric_initialization_preserves_original_failure(tmp_path, monkeypatch):
    sdk, run = Mock(), _run("a")
    sdk.init.return_value = run
    run.define_metric.side_effect = ValueError("invalid metric definition")
    run.finish.side_effect = OSError("SDK close failed")
    monkeypatch.setattr(wandb_utils, "wandb", sdk)
    with pytest.raises(ValueError, match="invalid metric definition"):
        wandb_utils.init_wandb_experiment(_args(tmp_path, "a"))
    run.finish.assert_called_once_with(exit_code=1)


def test_single_draft_still_uses_legacy_tracking(tmp_path, monkeypatch):
    run, writer = Mock(), Mock()
    monkeypatch.setattr("aspec.controller.loop.wandb.run", run)
    log = Mock()
    monkeypatch.setattr("aspec.controller.loop.wandb.log", log)
    monkeypatch.setattr("aspec.controller.loop.get_tb_writer", lambda: writer)
    _write_training_metrics({"train/avg_loss": 1.0}, 4, 5)
    log.assert_called_once_with({"train/avg_loss": 1.0})
    writer.add_scalar.assert_called_once_with("train/avg_loss", 1.0, 4)


@pytest.mark.parametrize("wandb_enabled", [False, True])
def test_single_draft_eval_reaches_tensorboard_event_file(tmp_path, monkeypatch, wandb_enabled):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path))
    monkeypatch.setattr(evaluation, "get_tb_writer", lambda: writer)
    monkeypatch.setattr(evaluation.wandb, "run", Mock() if wandb_enabled else None)
    log = Mock()
    monkeypatch.setattr(evaluation.wandb, "log", log)
    group = Mock()
    group.run_eval.return_value = [{"eval/avg_loss": 0.25, "eval/avg_acc": 0.5}]
    try:
        evaluation.run_eval(7, group, True)
        events = EventAccumulator(str(tmp_path)).Reload()
        for tag, value in (("eval/avg_loss", 0.25), ("eval/avg_acc", 0.5), ("eval/step", 7)):
            assert [(event.step, event.value) for event in events.Scalars(tag)] == [(7, value)]
        assert log.call_count == int(wandb_enabled)
    finally:
        writer.close()
