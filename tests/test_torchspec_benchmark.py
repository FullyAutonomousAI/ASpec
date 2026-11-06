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

"""Fast protocol checks; no model, datasets download, or CUDA execution."""

import json
from types import SimpleNamespace

import pytest
import torch

from tools.performance import torchspec_benchmark as bench


def test_dp_statistics_exclude_warmup_and_use_slowest_rank_wall():
    rows = [{
        "committed_documents": 4, "input_tokens": 100, "physical_tokens": 128,
        "supervised_predictions": 70, "anchor_slots": 2048, "query_anchor_slots": 2048,
        "configured_window_tokens": 128, "peak_allocated_bytes": 10 + rank,
        "peak_reserved_bytes": 20 + rank, "wall_seconds": 2 + rank,
        "warmup": 2, "accumulation_steps": 1, "batch_work": [{}] * 6,
    } for rank in range(2)]
    result = bench.aggregate_resident_results(rows)
    assert result["committed_documents"] == 8
    assert result["documents_per_physical_batch"] == 1
    assert result["input_tokens_per_second"] == 200 / 3
    assert result["padding_efficiency"] == 200 / 256
    assert result["peak_allocated_bytes"] == 11
    assert result["rank_results"] == rows
    assert "rank_results" not in rows[0]


def test_resident_dp_assignment_preserves_each_global_update():
    records = list(range(32))
    ranks = [bench.resident_rank_records(records, 16, 2, rank) for rank in range(2)]
    assert ranks == [list(range(0, 32, 2)), list(range(1, 32, 2))]
    for step in range(2):
        observed = [v for rank in ranks for v in rank[step * 8:(step + 1) * 8]]
        assert sorted(observed) == records[step * 16:(step + 1) * 16]
    # Splitting each fixed stream into different microbatch sizes cannot
    # move documents across optimizer updates or duplicate work.
    for mbs in (1, 2, 4, 8):
        for rank in ranks:
            batches = [rank[start:start + mbs] for start in range(0, len(rank), mbs)]
            assert [v for b in batches[:8 // mbs] for v in b] == rank[:8]


@pytest.mark.parametrize("count, batch, world, rank", [(15, 8, 2, 0), (16, 7, 2, 0), (16, 8, 2, 2)])
def test_resident_dp_assignment_rejects_incomplete_or_invalid_work(count, batch, world, rank):
    with pytest.raises(ValueError):
        bench.resident_rank_records(list(range(count)), batch, world, rank)


def test_nonpacking_cli_selects_native_per_document_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, "run_native", calls.append)
    monkeypatch.setattr(bench.sys, "argv", [
        "benchmark", "run", "--framework", "aspec", "--layout-mode", "non-packing",
        "--features", str(tmp_path / "features"), "--initial", str(tmp_path / "initial.pt"),
        "--result", str(tmp_path / "result.json"), "--documents-per-step", "1",
        "--micro-batch-size", "1",
    ])
    bench.main()
    assert len(calls) == 1
    assert calls[0].layout_mode == "non-packing"
    assert calls[0].micro_batch_size == calls[0].documents_per_step == 1


def test_cudnn_observer_preserves_auxiliary_outputs_and_records_thd(monkeypatch):
    from aspec.models.ops import cudnn_flex_attention as adapter

    def fake_attention(query, _key, _value, **_kwargs):
        return query * 2, torch.zeros(1)

    monkeypatch.setattr(adapter, "cudnn_flex_attention", fake_attention)
    observation = bench.observe_cudnn_attention()
    query = torch.ones(2, 3, 4, requires_grad=True)
    result = adapter.cudnn_flex_attention(query, query, query, return_lse=True)

    assert isinstance(result, tuple)
    result[0].sum().backward()
    assert observation == {
        "forward_calls": 1,
        "backward_calls": 1,
        "qkv_format": "thd",
    }


@pytest.mark.parametrize("explicit", [None, "2"])
def test_native_defaults_precede_package_import_and_preserve_caller(monkeypatch, tmp_path, explicit):
    package = tmp_path / "native_test_framework"
    (package / "utils").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('package imported before defaults')\n")
    (package / "utils/env.py").write_text(
        "import os\n"
        "def get_native_test_framework_env_vars():\n"
        "    return {'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '8'), 'HF_TOKEN': 'not-for-evidence'}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    if explicit is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", explicit)
    result = bench.configure_native_environment("native_test_framework")
    assert bench.os.environ["OMP_NUM_THREADS"] == (explicit or "8")
    assert result["cpu_thread_environment"]["OMP_NUM_THREADS"] == (explicit or "8")
    assert "not-for-evidence" not in json.dumps(result)
    assert "native_test_framework" not in __import__("sys").modules


def test_missing_native_environment_fails_closed():
    with pytest.raises(ModuleNotFoundError):
        bench.configure_native_environment("nonexistent_benchmark_framework_xyz")


def test_feature_context_only_extends_the_agreed_qwen3_8b_lane():
    text = SimpleNamespace(model_type="qwen3", max_position_embeddings=32768,
                           rope_parameters={"rope_theta": 1000000.0})
    assert bench.feature_target_config(text, 131072) is text
    assert text.max_position_embeddings == 131072
    assert text.rope_parameters["rope_type"] == "yarn"
    with pytest.raises(ValueError, match="128K"):
        bench.feature_target_config(text, 131073)
    text = SimpleNamespace(model_type="qwen3_5_text", max_position_embeddings=262144,
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7})
    outer = SimpleNamespace(text_config=text)
    assert bench.feature_target_config(outer, 197474) is outer
    assert text.max_position_embeddings == 262144
    assert text.rope_parameters == {"rope_type": "default", "rope_theta": 1e7}
    with pytest.raises(ValueError, match="native context"):
        bench.feature_target_config(outer, 262145)


def test_feature_capture_matches_logical_layer_output_not_its_input():
    import torch

    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + 10

    layer, captures = Layer(), {}
    handle = layer.register_forward_hook(bench.post_layer_capture(captures, 0))
    try:
        result = layer(torch.tensor([1.0, 2.0]))
        assert torch.equal(captures[0], result)
        assert torch.equal(captures[0], torch.tensor([11.0, 12.0]))
    finally:
        handle.remove()
    assert bench.feature_backbone(SimpleNamespace(layers=[layer])).layers == [layer]
    wrapper = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    assert bench.feature_backbone(wrapper).layers == [layer]
    with pytest.raises(ValueError, match="decoder"):
        bench.feature_backbone(SimpleNamespace())


def test_summary_counts_real_work():
    assert bench.summary([5, 1, 9]) == {
        "count": 3, "min": 1, "max": 9, "mean": 5.0,
        "p50": 5, "p90": 9, "p99": 9,
    }
    with pytest.raises(ValueError, match="eligible"):
        bench.summary([])


def test_optional_package_version_is_recorded_without_becoming_a_requirement(monkeypatch):
    assert bench.package_version("pytest") == pytest.__version__

    def missing(_distribution):
        raise bench.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(bench.importlib.metadata, "version", missing)
    assert bench.package_version("not-installed") is None


def test_selection_is_reproducible_without_duplicates(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(100)))
    actual = bench.reservoir(source, 12, 42)
    assert actual == bench.reservoir(source, 12, 42)
    assert len(actual) == len({i for i, _ in actual}) == 12
    assert [i for i, _ in actual] != list(range(12))
    assert actual != bench.reservoir(source, 12, 43)
    assert all(json.loads(row)["id"] == i for i, row in actual)


def test_small_source_is_not_duplicated_to_meet_a_requested_count(tmp_path):
    source = tmp_path / "small.jsonl"
    source.write_text("{}\n{}\n")
    assert len(bench.reservoir(source, 8, 42)) == 2


@pytest.mark.parametrize("mode", ["data", "features", "run"])
def test_missing_inputs_rejected_before_loading_models(monkeypatch, mode):
    monkeypatch.setattr("sys.argv", ["benchmark", mode])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_source_adapters_preserve_original_assistant_answer():
    answer = "Original answer, not regenerated.\n"
    assert bench.source_conversation({"prompt": "long context", "response": answer}) == [
        {"role": "user", "content": "long context"},
        {"role": "assistant", "content": answer},
    ]
    conversation = [{"from": "human", "value": "x"}, {"from": "gpt", "value": answer}]
    assert bench.source_conversation({"conversations": conversation}) is conversation
    assert bench.source_conversation({"messages": conversation}) is conversation
    assert bench.source_conversation({"prompt": "x"}) == []


def test_length_stratified_selection_is_deterministic_and_covers_endpoints():
    rows = [
        {"data_id": str(index), "seq_len": length}
        for index, length in enumerate((40, 10, 60, 20, 50, 30))
    ]
    selected = bench.length_stratified(rows, 4)
    assert [row["seq_len"] for row in selected] == [10, 30, 40, 60]
    assert bench.length_stratified(rows, 1)[0]["seq_len"] == 40
    with pytest.raises(ValueError, match="enough records"):
        bench.length_stratified(rows, 7)


def test_resident_feed_supports_both_native_iterator_apis():
    feed = bench.ResidentDocumentFeed(
        [{"value": 10}, {"value": 20}], [{"data_id": "a", "seq_len": 10}, {"data_id": "b", "seq_len": 20}],
        lambda items: items[0], with_identity=True,
    )
    assert iter(feed) is feed
    assert next(feed)["_aspec_work"] == {"work_id": "a", "execution_id": "resident-1"}
    rows = list(feed.iter_step_batches(2))
    assert [row["value"] for row in rows] == [20, 10]
    assert rows[-1]["_aspec_work"]["execution_id"] == "resident-3"
    assert "_aspec_work" not in feed.documents[0]


def test_multi_document_feed_keeps_every_original_identity():
    feed = bench.ResidentDocumentFeed(
        [{"value": 10}, {"value": 20}], [{"data_id": "a", "seq_len": 10}, {"data_id": "b", "seq_len": 20}],
        lambda items: {"items": items}, with_identity=True, documents_per_batch=2,
    )
    batch = next(feed)
    assert [v["_aspec_work"]["work_id"] for v in batch["items"]] == ["a", "b"]
    assert [v["value"] for v in batch["items"]] == [10, 20]
    assert batch["_aspec_benchmark_input_tokens"] == 30
    with pytest.raises(ValueError, match="aligned"):
        bench.ResidentDocumentFeed([], [], lambda x: x, with_identity=False)


def test_natural_packing_uses_bounded_pool_and_real_row_capacity():
    records = [
        {"data_id": str(index), "seq_len": length, "loss_tokens": length // 2}
        for index, length in enumerate((7, 6, 5, 4, 3, 2))
    ]
    schedule = bench.natural_packing_schedule(
        records, pool_size=4, max_seq_length=12, batches=3
    )
    assert len(schedule) == 3
    assert all(sum(records[index]["seq_len"] for index in row) <= 12 for row in schedule)
    assert all(row for row in schedule)
    assert any(len(row) > 1 for row in schedule)


def test_natural_packing_rejects_fake_capacity_and_incomplete_pool():
    records = [{"data_id": "a", "seq_len": 13, "loss_tokens": 4}]
    with pytest.raises(ValueError, match="fit the physical row"):
        bench.natural_packing_schedule(
            records, pool_size=1, max_seq_length=12, batches=1
        )
    with pytest.raises(ValueError, match="complete pool"):
        bench.natural_packing_schedule(
            records, pool_size=2, max_seq_length=16, batches=1
        )


def test_scheduled_feed_preserves_all_packed_document_identities():
    def collate(items):
        return {"items": items}

    feed = bench.ScheduledDocumentFeed(
        [{"value": 10}, {"value": 20}],
        [{"data_id": "a", "seq_len": 10}, {"data_id": "b", "seq_len": 20}],
        collate,
        [(1, 0)],
        with_identity=True,
    )
    batch = next(feed)
    assert batch["_aspec_benchmark_documents"] == 2
    assert batch["_aspec_benchmark_input_tokens"] == 30
    assert [item["value"] for item in batch["items"]] == [20, 10]
    assert [item["_aspec_work"]["work_id"] for item in batch["items"]] == ["b", "a"]


def test_cycled_document_occurrences_have_distinct_execution_identities():
    feed = bench.ScheduledDocumentFeed(
        [{"value": 10}], [{"data_id": "a", "seq_len": 10}], lambda items: {"items": items},
        [(0, 0)], with_identity=True,
    )
    first, second = next(feed), next(feed)
    identities = [item["_aspec_work"] for batch in (first, second) for item in batch["items"]]
    assert {item["work_id"] for item in identities} == {"a"}
    assert len({item["execution_id"] for item in identities}) == 4


def test_resident_counting_does_not_reduce_or_synchronize_a_gpu_mask():
    class NoMaskRead:
        def sum(self):
            raise AssertionError("Counting metadata must not launch a GPU reduction")

    batch = {"_aspec_benchmark_documents": 2, "_aspec_benchmark_input_tokens": 30,
             "input_ids": SimpleNamespace(numel=lambda: 32), "attention_mask": NoMaskRead()}
    assert bench.observed_batch_work(batch) == {"documents": 2, "input_tokens": 30, "physical_tokens": 32}
    assert not any(key.startswith("_aspec_benchmark") for key in batch)


def test_archive_source_revision_is_unknown_not_a_spurious_git_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=128, stdout=""))
    assert bench.source_revision(tmp_path / "package/__init__.py") is None
