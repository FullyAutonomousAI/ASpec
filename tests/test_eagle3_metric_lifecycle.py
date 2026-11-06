# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Existing EAGLE report lifecycle and independent correctness fixes."""
from unittest.mock import Mock

import pytest
import torch

from aspec.training.eagle3_trainer import Eagle3Trainer
from aspec.training.trainer import DeferredTrainMetrics


def trainer_for():
    trainer = object.__new__(Eagle3Trainer)
    trainer._pending_train_metrics = None
    return trainer


def deferred(step, events):
    event = Mock()
    event.synchronize.side_effect = lambda: events.append(f"ready:{step}")
    return DeferredTrainMetrics(torch.tensor([float(step)]), torch.tensor([float(step)]), event, step,
                                lambda values: {"train/avg_loss": values[0]})


def test_one_step_delay_resume_and_final_flush_are_not_replayed():
    trainer = trainer_for()
    trainer.data_fetcher = object()
    trainer._enable_perf_metrics = False
    trainer.prof = Mock()
    events = []
    def core(step, num_batches):
        events.append(f"submit:{step}")
        return deferred(step, events)
    trainer._train_core_from_queue = core
    assert trainer.train_from_queue(9, 1) == {}
    assert trainer.train_from_queue(10, 1)["_metrics_step"] == 9
    assert events == ["submit:9", "submit:10", "ready:9"]
    assert trainer.flush_pending_metrics()["_metrics_step"] == 10
    assert trainer.flush_pending_metrics() == {}


def test_failed_finalize_retains_pending_for_explicit_retry():
    trainer = trainer_for()
    pending = Mock()
    pending.finalize.side_effect = [RuntimeError("copy failed"), {"_metrics_step": 9}]
    trainer._pending_train_metrics = pending
    with pytest.raises(RuntimeError, match="copy failed"):
        trainer.flush_pending_metrics()
    assert trainer._pending_train_metrics is pending
    assert trainer.flush_pending_metrics() == {"_metrics_step": 9}
    assert trainer._pending_train_metrics is None


def test_failed_next_step_does_not_advance_pending_metric_identity():
    trainer = trainer_for()
    trainer.data_fetcher = object()
    trainer._enable_perf_metrics = False
    trainer.prof = Mock()
    trainer._pending_train_metrics = deferred(9, [])
    trainer._train_core_from_queue = Mock(side_effect=RuntimeError("backward failed"))
    with pytest.raises(RuntimeError, match="backward failed"):
        trainer.train_from_queue(10, 1)
    assert trainer._pending_train_metrics.metric_step == 9
    assert trainer.flush_pending_metrics()["_metrics_step"] == 9


def test_deferred_timing_uses_own_step_events_after_completion():
    value = deferred(9, [])
    a, b = Mock(), Mock()
    a.elapsed_time.return_value = 1250.
    value.step_events = (a, b)
    value.compute_events = [(a, b)]
    value.optimizer_events = [(a, b)]
    out = value.finalize()
    b.synchronize.assert_called_once()
    value.ready_event.synchronize.assert_not_called()
    assert out["perf/step_time"] == out["perf/compute_time"] == out["perf/optimizer_time"] == 1.25
    assert out["_metrics_step"] == 9



def test_eagle_delayed_marker_uses_driver_step_after_resume():
    import os
    from argparse import Namespace
    from unittest.mock import patch
    if os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1":
        pytest.skip("explicit GPU integration gate")
    trainer = trainer_for()
    trainer.global_step = 7
    trainer.optimizer = Mock()
    trainer.optimizer.get_learning_rate.return_value = .001
    trainer.args = Namespace(ploss_weights=None)
    trainer._ploss_weights = (1., 1.)
    trainer._ploss_weight_sum = 2.
    trainer._metric_buffers = []
    metrics = [{k: torch.tensor(v, device="cuda") for k, v in {
        "vlosses": [1., 2.], "acces": [.5, .25],
        "acc_counts": [3., 2.], "alphas": [.4, .2],
    }.items()}]
    with patch("aspec.training.eagle3_trainer.dist.all_reduce"):
        report = trainer._aggregate_metrics(metrics, step=6, grad_norm=torch.tensor(3., device="cuda"))
    assert isinstance(report, DeferredTrainMetrics)
    assert report.metric_step == 6
    trainer.global_step = 8
    with patch("aspec.training.eagle3_trainer.dist.get_rank", return_value=0):
        result = report.finalize()
    assert result["_metrics_step"] == 6
    assert result["train/global_step"] == 7
    assert result["train/lr"] == .001
