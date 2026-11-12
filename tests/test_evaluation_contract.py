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
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from aspec.controller.eval import eval_cache_identity
from aspec.controller.loop import _cleanup_old_checkpoints
from aspec.training.evaluation import evaluation_rng, training_attention_evidence
from aspec.training.trainer import Trainer


def test_eval_forward_cannot_erase_training_backward_evidence_on_error():
    draft = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    draft[0].attention_observation = {"backward_observed": True, "qkv_format": "thd"}
    original = draft[0].attention_observation
    with pytest.raises(RuntimeError), training_attention_evidence(draft):
        draft[0].attention_observation.update(backward_observed=False)
        draft[1].attention_observation = {"backward_observed": False}
        raise RuntimeError("eval failed")
    assert draft[0].attention_observation is original
    assert original == {"backward_observed": True, "qkv_format": "thd"}
    assert not hasattr(draft[1], "attention_observation")


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_eval_rebatch_preserves_document_length_and_local_cp_ownership(cp_size):
    from aspec.data.utils import DataCollatorWithPadding
    from aspec.inference.context import context_shard_bounds
    from aspec.models.ops.eagle_thd import EagleTHDLayout

    for rank in range(cp_size):
        lo, hi = context_shard_bounds(130, rank, cp_size)
        collator = DataCollatorWithPadding(direct_cp_enabled=cp_size > 1)
        sample = {"input_ids": torch.arange(130).unsqueeze(0),
                  "loss_mask": torch.ones(1, 130),
                  "hidden_states": torch.randn(1, hi - lo, 12),
                  "last_hidden_states": torch.randn(1, hi - lo, 4)}
        cached = collator([sample])
        replayed = collator([cached])
        assert replayed["attention_mask"].sum().item() == 130
        layout, auxiliary, teacher = EagleTHDLayout.from_batch(
            replayed, cp_rank=rank, cp_size=cp_size,
        )
        assert layout.lengths == (130,)
        assert layout.owned == hi - lo
        torch.testing.assert_close(auxiliary[:, :hi - lo], sample["hidden_states"])
        torch.testing.assert_close(teacher[:, :hi - lo], sample["last_hidden_states"])


@pytest.mark.parametrize("metric_key", ["eval/simulated_acc_len", "eval/suffix_simulated_acc_len"])
def test_exact_resume_retains_eval_history_only_through_selected_frontier(tmp_path, monkeypatch, metric_key):
    from aspec.controller import eval as evaluation

    target = tmp_path / "target"
    target.mkdir()
    previous = tmp_path / "previous"
    checkpoint = previous / "checkpoints/iter_0000009"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").touch()
    (checkpoint / "meta.json").write_text('{"global_step": 8}')
    later = previous / "checkpoints/iter_0000017"
    later.mkdir()
    args = SimpleNamespace(eval_interval=1, dp_size=1, checkpoint_dir=None,
                           target_model_path=str(target), inference_batch_size=1,
                           load_path=str(checkpoint))
    identity = evaluation.eval_cache_identity(args, "sha256:eval", 2)
    (previous / "evaluation.json").write_text(json.dumps({"cache_identity": identity, "history": [
        {"step": 8, "checkpoint": str(checkpoint), "metrics": {metric_key: 0.3}},
        {"step": 16, "checkpoint": str(later), "metrics": {metric_key: 0.9}},
    ]}))
    controller, group = Mock(), Mock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    group.load_eval_cache.return_value = [2]
    monkeypatch.setattr(evaluation.ray, "get", lambda value: value)
    state = evaluation.setup_eval(controller, group, args, 2)
    assert [r["step"] for r in state.history] == [8]
    assert state.best_eval_score == 0.3


def test_shared_eval_budget_is_installed_before_source_admission(monkeypatch):
    from aspec import multi_draft
    from aspec.controller.eval import EvalSetupState

    events = []
    hub, manager, group, controller = (Mock() for _ in range(4))
    monkeypatch.setattr(multi_draft.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(multi_draft, "ExperimentController", lambda *_: controller)
    controller.get_eval_dataset_size.remote.return_value = 2
    controller.try_dispatch_eval_batch.remote.return_value = True
    state = EvalSetupState(1, True, False, "cache", -float("inf"), 1, 2, 1)
    monkeypatch.setattr(multi_draft, "setup_eval", lambda *_: state)
    manager.start_eval_budget.remote.side_effect = lambda *_: events.append("budget")
    hub.begin_eval.remote.side_effect = lambda *_: events.append("source") or 2
    result = multi_draft.prepare_shared_eval(hub, manager, {"a": group}, {"a": SimpleNamespace()})
    assert events == ["budget", "source"]
    assert result["a"].eval_cache_loaded
    assert group.cache_eval_samples.call_count == 2
    hub.end_eval.remote.assert_called_once()


def test_partial_rank_cache_is_cleared_before_regeneration(tmp_path, monkeypatch):
    from aspec.controller import eval as evaluation

    controller, group = Mock(), Mock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:eval"}}
    group.load_eval_cache.return_value = [2, 0]
    monkeypatch.setattr(evaluation.ray, "get", lambda value: value)
    args = SimpleNamespace(eval_interval=1, dp_size=2, checkpoint_dir=None,
                           target_model_path=str(tmp_path), inference_batch_size=2)
    state = evaluation.setup_eval(controller, group, args, 4)
    assert not state.eval_cache_loaded
    group.clear_eval_cache.assert_called_once()


def test_retention_protects_best_weights_not_just_pointer(tmp_path):
    for step in (2, 3, 4):
        (tmp_path / f"iter_{step:07d}").mkdir()
    (tmp_path / "best_checkpointed_iteration.txt").write_text("2")
    _cleanup_old_checkpoints(str(tmp_path), 1)
    assert (tmp_path / "iter_0000002").is_dir()
    assert not (tmp_path / "iter_0000003").exists()
    assert (tmp_path / "iter_0000004").is_dir()


def test_cache_identity_changes_with_processed_data_layout_and_target(tmp_path):
    args = SimpleNamespace(target_model_path=str(tmp_path), draft_context_parallel_size=1)
    first = eval_cache_identity(args, "sha256:one", 4)
    assert first != eval_cache_identity(args, "sha256:two", 4)
    args.draft_context_parallel_size = 2
    assert first != eval_cache_identity(args, "sha256:one", 4)
    args.draft_context_parallel_size = 1
    (tmp_path / "config.json").write_text('{"rope_theta": 10000}')
    assert first != eval_cache_identity(args, "sha256:one", 4)


def test_cache_identity_covers_target_quantization_and_parallel_settings(tmp_path):
    args = SimpleNamespace(target_model_path=str(tmp_path), sglang_extra_args={})
    initial = eval_cache_identity(args, "sha256:one", 4)
    args.sglang_extra_args = {"quantization": "fp8"}
    assert initial != eval_cache_identity(args, "sha256:one", 4)
    args.sglang_extra_args = {}
    args.sglang_tp_size = 2
    assert initial != eval_cache_identity(args, "sha256:one", 4)


def test_preprocessing_identity_covers_content_even_if_stat_is_preserved(tmp_path):
    import os

    from aspec.data.dataset import _preprocessing_identity

    source = tmp_path / "data.jsonl"
    source.write_text("first")
    args = SimpleNamespace(train_data_path=str(source), target_model_path=str(tmp_path))
    before = _preprocessing_identity(args, [], None)
    stat = source.stat()
    source.write_text("other")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert before != _preprocessing_identity(args, [], None)
    before = _preprocessing_identity(args, [], None)
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "changed"}')
    assert before != _preprocessing_identity(args, [], None)


def test_bound_cache_rejects_legacy_and_mismatched_payload(tmp_path):
    trainer = SimpleNamespace(_wait_for_eval_cache_save=Mock(), cache_rank=0, dp_rank=0)
    path = tmp_path / "eval_rank_0.pt"
    torch.save([{"input_ids": torch.ones(1, 2)}], path)
    assert Trainer.load_eval_cache(trainer, str(tmp_path), identity={"data": 1}) == 0
    torch.save({"identity": {"data": 1}, "samples": [{"input_ids": torch.ones(1, 2)}]}, path)
    assert Trainer.load_eval_cache(trainer, str(tmp_path), identity={"data": 1}) == 1
    assert Trainer.load_eval_cache(trainer, str(tmp_path), identity={"data": 2}) == 0


def test_identical_consumers_can_publish_same_eval_cache_concurrently(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    rendezvous = Barrier(2)
    original_save = torch.save

    def synchronized_save(payload, destination):
        original_save(payload, destination)
        rendezvous.wait(timeout=2)

    monkeypatch.setattr(torch, "save", synchronized_save)
    with ThreadPoolExecutor(2) as executor:
        trainers = [SimpleNamespace(
            _eval_cache=[{"input_ids": torch.ones(1, 2)}], cache_rank=0,
            _eval_cache_identity={"data": 1}, _wait_for_eval_cache_save=Mock(),
            _io_executor=executor,
        ) for _ in range(2)]
        for trainer in trainers:
            Trainer.save_eval_cache(trainer, str(tmp_path))
        for trainer in trainers:
            trainer._eval_cache_save_future.result(timeout=5)
    payload = torch.load(tmp_path / "eval_rank_0.pt", weights_only=False)
    assert payload["identity"] == {"data": 1}
    torch.testing.assert_close(payload["samples"][0]["input_ids"], torch.ones(1, 2))
    assert not list(tmp_path.glob("*.tmp"))


def test_requested_eval_cannot_silently_disappear_after_dp_truncation():
    from aspec.controller.training_controller import TrainingControllerState

    state = SimpleNamespace(dp_size=8, _load_dataset_split=lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="no complete DP batch"):
        TrainingControllerState.load_eval_dataset(state, SimpleNamespace(eval_data_path="requested"))


def test_eval_rng_is_repeatable_and_restores_training_stream_on_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(23)
    np.random.seed(24)
    torch.manual_seed(25)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with evaluation_rng(5):
        expected = (random.random(), np.random.rand(), torch.rand(4))
    with pytest.raises(RuntimeError), evaluation_rng(5):
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
        raise RuntimeError("eval failed")
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_eval_cache_binds_the_current_sglang_kernel_distribution(tmp_path, monkeypatch):
    from aspec.controller import eval as evaluation

    kernel = {"version": "0.4.7"}
    monkeypatch.setattr(evaluation.importlib.metadata, "version",
                        lambda name: kernel["version"] if name == "sglang-kernel" else "fixed")
    args = SimpleNamespace(target_model_path=str(tmp_path))
    first = evaluation.eval_cache_identity(args, "sha256:data", 4)
    assert first["versions"]["sglang-kernel"] == "0.4.7"
    kernel["version"] = "0.4.8"
    assert first != evaluation.eval_cache_identity(args, "sha256:data", 4)

@pytest.mark.parametrize("extension", ["jsonl", "parquet"])
def test_directory_cache_identity_tracks_shard_content_and_membership(tmp_path, extension):
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    from aspec.data.dataset import _preprocessing_identity
    from aspec.data.utils import load_hf_dataset
    source = tmp_path / "data"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    def write(path, value):
        if extension == "jsonl":
            path.write_text(json.dumps({"text": value}) + "\n")
        else:
            pq.write_table(pa.table({"text": [value]}), path)
    first = source / f"first.{extension}"
    write(first, "before")
    dataset = load_hf_dataset(str(source))
    args = SimpleNamespace(train_data_path=str(source), target_model_path=str(target))
    def identity():
        return _preprocessing_identity(args, dataset, None)
    original = identity()
    stamp = first.stat()
    write(first, "after!")
    os.utime(first, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert identity() != original
    changed = identity()
    second = source / f"second.{extension}"
    write(second, "extra")
    assert identity() != changed
    added = identity()
    second.rename(source / f"renamed.{extension}")
    assert identity() != added
    (source / f"renamed.{extension}").unlink()
    assert identity() == changed
    (source / "notes.txt").write_text("not a dataset shard")
    assert identity() == changed
