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

from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.performance.torchspec_e2e_benchmark as benchmark_module
from tools.performance.torchspec_e2e_benchmark import (
    NativeObserver,
    _path_identity,
    batch_work_ids,
    combine_rank_metrics,
    count_batch_documents,
    worker_pythonpath,
)


@pytest.mark.parametrize("framework", ["aspec", "torchspec", "angelspec"])
def test_observer_uses_native_framework_entry(framework):
    seen = []
    entry = SimpleNamespace(
        main=lambda: seen.append("main"),
        parse_config=lambda: "native-config",
        train_async_no_generation=lambda config: seen.append(config),
    )
    benchmark_module.invoke_native_entry(framework, entry)
    assert seen == ["main" if framework == "aspec" else "native-config"]


def row(step=1, rank=0):
    return {"_benchmark": {"step": step, "rank": rank, "documents": 2,
                          "input_tokens": 123, "supervised_predictions": 77,
                          "anchor_slots": 512,
                          "physical_tokens": 128,
                          "cudnn_calls": {"forward_calls": 5, "backward_calls": 5},
                          "runtime": {"block_size": 7},
                          "loss": 1.0, "grad_norm": 2.0}}


def test_counts_are_summed_from_every_rank_not_multiplied_from_rank_zero():
    second = row(rank=1)
    second["_benchmark"]["input_tokens"] = 999
    combined = combine_rank_metrics([row(), second])
    assert combined["input_tokens"] == 1122
    assert combined["documents"] == 4
    assert combined["supervised_predictions"] == 154
    assert combined["anchor_slots"] == 1024
    assert combined["block_size"] == 7


def test_document_count_uses_source_work_instead_of_packed_rows():
    packed = {
        "input_ids": SimpleNamespace(shape=(1, 4096)),
        "_aspec_work": [
            {"work_id": str(index), "execution_id": str(index)}
            for index in range(7)
        ],
    }
    assert count_batch_documents(packed) == 7
    assert count_batch_documents({"input_ids": SimpleNamespace(shape=(3, 4096))}) == 3
    with pytest.raises(ValueError, match="non-empty"):
        count_batch_documents({"input_ids": SimpleNamespace(shape=(1, 1)), "_aspec_work": []})
    assert batch_work_ids(packed) == [str(index) for index in range(7)]


def cp_rows(dp_size=1, cp_size=2):
    rows = []
    for dp_rank in range(dp_size):
        for cp_rank in range(cp_size):
            record = row(rank=dp_rank * cp_size + cp_rank)
            record["_benchmark"].update(
                dp_rank=dp_rank, cp_rank=cp_rank, cp_size=cp_size,
                work_ids=[f"{dp_rank}-a", f"{dp_rank}-b"],
            )
            rows.append(record)
    return rows


@pytest.mark.parametrize("dp_size,cp_size", [(1, 2), (2, 2), (1, 4)])
def test_cp_counts_source_work_once_and_sums_disjoint_supervision(dp_size, cp_size):
    records = cp_rows(dp_size, cp_size)
    result = combine_rank_metrics(records, expected_ranks=set(range(dp_size * cp_size)))
    assert result["documents"] == 2 * dp_size
    assert result["input_tokens"] == 123 * dp_size
    assert result["physical_tokens"] == 128 * dp_size
    assert result["anchor_slots"] == 512 * dp_size
    assert result["supervised_predictions"] == 77 * dp_size * cp_size
    assert result["cudnn_calls"]["forward_calls"] == 5 * dp_size * cp_size
    assert len(result["work_ids"]) == len(set(result["work_ids"])) == 2 * dp_size


def test_cp_incomplete_group_fails_closed():
    with pytest.raises(ValueError, match="context parallel rank"):
        combine_rank_metrics(cp_rows()[:1])


@pytest.mark.parametrize("field,value", [("work_ids", ["wrong"]), ("input_tokens", 999)])
def test_cp_mismatched_source_work_fails_closed(field, value):
    records = cp_rows()
    records[1]["_benchmark"][field] = value
    with pytest.raises(ValueError, match="source work"):
        combine_rank_metrics(records)


def test_invalid_rank_step_and_nonfinite_loss_fail_closed():
    for records in ([row(), row()], [row(), row(step=2, rank=1)]):
        with pytest.raises(ValueError):
            combine_rank_metrics(records)
    broken = row()
    broken["_benchmark"]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        combine_rank_metrics([broken])


@pytest.mark.parametrize("ranks", [[0], [0, 2]])
def test_missing_or_unexpected_trainer_ranks_fail_closed(ranks):
    with pytest.raises(ValueError, match="trainer ranks"):
        combine_rank_metrics([row(rank=rank) for rank in ranks], expected_ranks={0, 1})


def test_incomplete_configured_window_is_not_a_successful_benchmark():
    observer = NativeObserver(SimpleNamespace(get=lambda value: value), warmup=1)
    observer.expected_steps = 3
    observer.records = [{"step": 1}, {"step": 2}]
    with pytest.raises(ValueError, match="Incomplete configured"):
        observer.summarize()


def test_native_loop_binds_expected_ranks_and_optimizer_steps():
    observer = NativeObserver(SimpleNamespace(get=lambda value: value), warmup=1)
    controller = SimpleNamespace(get_full_status=SimpleNamespace(remote=lambda: {}))
    inference = SimpleNamespace(get_status=SimpleNamespace(remote=lambda: {}))

    def loop(*args, **kwargs):
        for step in (1, 2):
            observer.record([row(step, rank=0), row(step, rank=1)])
        return "native-result"

    wrapped = observer.wrap_loop(loop)
    result = wrapped(
        SimpleNamespace(num_train_steps=2), controller, inference,
        SimpleNamespace(_actor_handlers=[object(), object()]),
    )
    assert result == "native-result"
    assert observer.expected_steps == 2
    assert observer.expected_ranks == {0, 1}
    assert not observer.active
    assert observer.summarize()["measured_steps"] == 1


@pytest.mark.parametrize("end", [10.0, 9.0, float("nan")])
def test_invalid_completion_clock_is_rejected(end):
    observer = NativeObserver(SimpleNamespace(get=lambda value: value), warmup=1)
    observer.records = [{"step": 1, "completed_at": 10.0}, {"step": 2, "completed_at": end}]
    with pytest.raises(ValueError, match="completion clock"):
        observer.summarize()


def test_nonfinite_interior_completion_is_rejected():
    observer = NativeObserver(SimpleNamespace(get=lambda value: value), warmup=1)
    observer.records = [
        {"step": 1, "completed_at": 10.0},
        {"step": 2, "completed_at": float("nan")},
        {"step": 3, "completed_at": 20.0},
    ]
    with pytest.raises(ValueError, match="completion clock"):
        observer.summarize()


def test_observer_preserves_results_and_continuous_completion_clock(monkeypatch):
    clock = iter([10.0, 15.0, 22.0, 23.0])
    monkeypatch.setattr("tools.performance.torchspec_e2e_benchmark.time.perf_counter", lambda: next(clock))
    ray = SimpleNamespace(get=lambda value: value)
    observer = NativeObserver(ray, warmup=1)
    observer.controller = SimpleNamespace(get_full_status=SimpleNamespace(remote=lambda: {"pool": 3}))
    observer.active = True
    observer.started_at, observer.finished_at = 1.0, 25.0
    for step in (1, 2, 3):
        results = [row(step), row(step, rank=1)]
        assert observer.get(results) is results
    summary = observer.summarize()
    assert summary["wall_seconds"] == 12
    assert summary["documents"] == 8
    assert summary["input_tokens"] == 492
    assert summary["supervised_predictions"] == 308
    assert summary["anchor_slots"] == 2048
    assert summary["physical_tokens"] == 512
    assert summary["padding_efficiency"] == pytest.approx(492 / 512)
    assert summary["cudnn_calls"] == {"forward_calls": 20, "backward_calls": 20}
    assert summary["draft_block_size"] == 7
    assert summary["supervision_capacity_positions"] == 14336
    assert summary["supervision_fill_ratio"] == pytest.approx(308 / 14336)
    assert summary["full_corpus"] == {
        "documents": 12,
        "input_tokens": 738,
        "supervised_predictions": 462,
        "anchor_slots": 3072,
        "physical_tokens": 768,
        "optimizer_steps": 3,
        "time_to_last_optimizer_completion_seconds": 21,
        "documents_per_second": pytest.approx(12 / 21),
        "input_tokens_per_second": pytest.approx(738 / 21),
        "supervised_predictions_per_second": pytest.approx(462 / 21),
        "padding_efficiency": pytest.approx(738 / 768),
        "unique_work_ids": None,
        "work_ids_sha256": None,
    }
    assert observer.boundaries["warmup"] == {"pool": 3}
    with pytest.raises(ValueError, match="Duplicate"):
        observer.get([row(3)])


def test_finite_source_observer_accepts_bounded_drop_last_from_training_evidence(monkeypatch):
    clock = iter([10.0, 15.0, 22.0, 23.0])
    monkeypatch.setattr(
        "tools.performance.torchspec_e2e_benchmark.time.perf_counter",
        lambda: next(clock),
    )
    ray = SimpleNamespace(get=lambda value: value)
    observer = NativeObserver(
        ray,
        warmup=1,
        expected_requested_documents=14,
        max_dropped_documents=2,
    )
    observer.controller = SimpleNamespace(
        get_full_status=SimpleNamespace(remote=lambda: {"pool": 3})
    )
    observer.active = True
    observer.started_at, observer.finished_at = 1.0, 25.0
    for step in (1, 2, 3):
        observer.get([row(step), row(step, rank=1)])
    observer.training_result = {
        "requested_source_work_budget": 14,
        "effective_source_work": 12,
        "committed_source_work": 12,
        "dropped_source_work": 2,
        "source_work_tail_policy": "drop_last_complete_packed_optimizer_step",
    }

    assert observer.summarize()["finite_source"] == {
        "requested_documents": 14,
        "effective_documents": 12,
        "dropped_documents": 2,
        "max_dropped_documents": 2,
        "tail_policy": "drop_last_complete_packed_optimizer_step",
    }


def test_finite_source_observer_rejects_mismatched_or_excessive_drop_last(monkeypatch):
    clock = iter([10.0, 15.0, 22.0, 23.0])
    monkeypatch.setattr(
        "tools.performance.torchspec_e2e_benchmark.time.perf_counter",
        lambda: next(clock),
    )
    observer = NativeObserver(
        SimpleNamespace(get=lambda value: value),
        warmup=1,
        expected_requested_documents=14,
        max_dropped_documents=1,
    )
    observer.controller = SimpleNamespace(
        get_full_status=SimpleNamespace(remote=lambda: {"pool": 3})
    )
    observer.active = True
    observer.started_at, observer.finished_at = 1.0, 25.0
    for step in (1, 2, 3):
        observer.get([row(step), row(step, rank=1)])
    observer.training_result = {
        "requested_source_work_budget": 14,
        "effective_source_work": 12,
        "committed_source_work": 12,
        "dropped_source_work": 2,
        "source_work_tail_policy": "drop_last_complete_packed_optimizer_step",
    }

    with pytest.raises(ValueError, match="bound exceeded"):
        observer.summarize()


def test_rank_block_size_mismatch_fails_closed():
    second = row(rank=1)
    second["_benchmark"]["runtime"] = {"block_size": 5}
    with pytest.raises(ValueError, match="block size"):
        combine_rank_metrics([row(), second])


@pytest.mark.parametrize("identity_changes", [False, True])
@pytest.mark.parametrize("cpu_microbatches", [0, 1, 2])
def test_actor_counter_growth_is_not_a_runtime_identity_change(
    monkeypatch, identity_changes, cpu_microbatches
):
    import torch

    import tools.performance.torchspec_benchmark as resident_module

    class FakeTrainer:
        def _train_step(self, batch):
            self.model.l1_kernel_invocations["indexed_tiled_fp32"] += 2
            return {"count_per_position": torch.ones(7)}

        def _aggregate_metrics(self, metrics, step, **kwargs):
            return {"train/avg_loss": 1.0, "train/grad_norm": 2.0}

        def _iter_batches_from_queue(self):
            return iter(())

    class FakePaddedCollator:
        def __call__(self, features):
            return {}

    monkeypatch.setenv("ASPEC_BENCH_NAMESPACE", "aspec")
    # Exercise the real observer with a fake native trainer, without importing
    # the GPU training stack or compiling a model during an offline unit test.
    project = Path(benchmark_module.__file__).resolve().parents[2]
    fake_modules = {
        "aspec": SimpleNamespace(__file__=str(project / "aspec/__init__.py")),
        "aspec.data.utils": SimpleNamespace(DataCollatorWithPadding=FakePaddedCollator),
        "aspec.models.dspark": SimpleNamespace(__file__=str(project / "aspec/models/dspark.py")),
        "aspec.training.dspark_trainer": SimpleNamespace(
            DSparkTrainer=FakeTrainer, __file__=str(project / "aspec/training/dspark_trainer.py"),
        ),
    }
    original_import = benchmark_module.importlib.import_module
    monkeypatch.setattr(benchmark_module.importlib, "import_module", lambda name:
                        fake_modules[name] if name in fake_modules else original_import(name))
    monkeypatch.setattr(resident_module, "observe_cudnn_attention", lambda: {
        "forward_calls": 0, "backward_calls": 0, "qkv_format": "thd",
    })
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test-gpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    benchmark_module.install_actor_observers()
    trainer = FakeTrainer()
    trainer.args = SimpleNamespace(anchor_sampling_scope="sample")
    trainer.model = SimpleNamespace(l1_loss_backend="tiled_fp32",
                                    l1_kernel_invocations={"indexed_tiled_fp32": 0},
                                    backbone_blocks_observed=3, loss_head_blocks_observed=2)
    trainer.draft_model = SimpleNamespace(layers=[])
    trainer.optimizer = SimpleNamespace(_fuse_clip_scaling=False)
    trainer.num_anchors, trainer.block_size, trainer.dp_rank = 512, 7, 0
    metrics = [trainer._train_step({
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        **({"_benchmark_input_tokens_cpu": 3} if index < cpu_microbatches else {}),
    }) for index in range(2)]
    materializations = []
    original_item = torch.Tensor.item

    def observed_item(tensor):
        materializations.append(tensor)
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", observed_item)
    assert [m["_benchmark_l1_kernel_invocations"]["indexed_tiled_fp32"] for m in metrics] == [2, 4]
    if identity_changes:
        metrics[1]["_benchmark_runtime"]["gpu_name"] = "changed-gpu"
        with pytest.raises(RuntimeError, match="runtime identity changed"):
            trainer._aggregate_metrics(metrics, 0)
    else:
        combined = trainer._aggregate_metrics(metrics, 0)["_benchmark"]
        assert combined["l1_kernel_invocations"] == {"indexed_tiled_fp32": 4}
        assert "l1_kernel_invocations" not in combined["runtime"]
        assert combined["query_shapes"] == {"backbone_slots": 6, "head_slots": 4}
        assert combined["input_tokens"] == 6
        # Prediction count plus at most one fallback token-count read.
        assert len(materializations) == (1 if cpu_microbatches == 2 else 2)


@pytest.mark.parametrize("pre_padded,usp", [(False, False), (True, False), (False, True)])
def test_collator_input_count_uses_only_known_unpadded_shapes(pre_padded, usp):
    class Collator:
        usp_enabled = usp

        def __call__(self, features):
            return {"native": True}

    benchmark_module.observe_padded_collator_inputs(Collator)
    samples = [{"input_ids": SimpleNamespace(shape=(1, length))} for length in (13, 29)]
    if pre_padded:
        samples[0]["attention_mask"] = object()
    result = Collator()(samples)
    assert result["native"]
    assert result.get("_benchmark_input_tokens_cpu") == (42 if not pre_padded and not usp else None)


def test_per_rank_native_timings_are_preserved_without_mutating_results():
    first, second = row(), row(rank=1)
    first.update({"perf/step_time": 2.0, "perf/data_time": .1})
    second.update({"perf/step_time": 3.0, "perf/data_time": .5})
    result = combine_rank_metrics([first, second])
    assert [r["native_timing"]["perf/step_time"] for r in result["ranks"]] == [2.0, 3.0]
    assert "native_timing" not in first["_benchmark"]


def test_input_file_identity_is_content_addressed(tmp_path):
    data = tmp_path / "documents.parquet"
    data.write_bytes(b"real benchmark input")
    identity = _path_identity(data)
    assert identity["kind"] == "file"
    assert identity["size_bytes"] == len(b"real benchmark input")
    assert len(identity["sha256"]) == 64
    assert _path_identity(tmp_path / "missing") == {
        "path": str((tmp_path / "missing").resolve()),
        "exists": False,
    }


def test_ray_worker_pythonpath_includes_checkout_and_preserves_existing_entries():
    result = worker_pythonpath({"PYTHONPATH": "/custom/one:/custom/two"}).split(":")
    assert result[0] == str(Path(benchmark_module.__file__).resolve().parents[2])
    assert result[1:] == ["/custom/one", "/custom/two"]
