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

import io
import json
from pathlib import Path

import pytest

from tools.performance import specforge_e2e_benchmark as benchmark
from tools.performance.specforge_e2e_benchmark import (
    _native_train_command,
    _parse_metrics,
    prepare_data,
    prepare_initial,
    summarize_metrics,
    unpack_loss_mask,
)


def test_native_command_resolves_config_before_switching_checkout(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    config.write_text("model: {}\n")
    monkeypatch.chdir(tmp_path.parent)

    command = _native_train_command(
        Path(tmp_path.name) / config.name,
        ["training.max_steps=1"],
    )

    assert command[-2] == str(config.resolve())
    assert command[-1] == "training.max_steps=1"


@pytest.mark.parametrize("ipc", [None, "1"])
def test_run_records_nvlink_allocator_enablement(tmp_path, monkeypatch, ipc):
    config = tmp_path / "benchmark.yaml"
    config.write_text("model: {}\n")
    args = benchmark.build_parser().parse_args([
        "run", "--checkout", str(tmp_path), "--config", str(config),
        "--result", str(tmp_path / "result.json"),
        "--log", str(tmp_path / "native.log"),
        "--expected-steps", "2", "--expected-ranks", "1",
        "--expected-documents", "2", "--expected-input-tokens", "4",
        "--expected-supervised-positions", "2", "--warmup-steps", "1",
        "--native-timers-only",
    ])
    monkeypatch.delenv("MC_USE_NVLINK_IPC", raising=False)
    if ipc is not None:
        monkeypatch.setenv("MC_USE_NVLINK_IPC", ipc)
    monkeypatch.setenv("MC_FORCE_MNNVL", "1")

    class CompletedProcess:
        stdout = io.StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **k: CompletedProcess())
    monkeypatch.setattr(benchmark, "summarize_metrics", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "_source_identity", lambda *a: {})
    monkeypatch.setattr(benchmark, "_system_identity", lambda: {})

    result = benchmark.run(args)

    assert result["environment"]["MC_USE_NVLINK_IPC"] == ipc
    assert result["environment"]["MC_FORCE_MNNVL"] == "1"
    assert json.loads(args.result.read_text())["environment"] == result["environment"]


@pytest.mark.parametrize(
    "name",
    [
        "specforge_qwen3_8b.yaml",
        "specforge_qwen3_8b_128k.yaml",
    ],
)
def test_managed_local_configs_use_node_local_control_directory(name):
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "benchmarks" / name).read_text(encoding="utf-8")

    assert "control_dir: /tmp/" in text


def test_parse_metrics_accepts_concatenated_rank_output():
    metrics = _parse_metrics(
        [
            "\x1b[32mstep 1: {'perf/optimizer_step_time_s': 1.0}\x1b[0m"
            "step 1: {'perf/optimizer_step_time_s': 1.1}",
            "unrelated log line",
        ]
    )

    assert metrics == {
        1: [
            {"perf/optimizer_step_time_s": 1.0},
            {"perf/optimizer_step_time_s": 1.1},
        ]
    }


def test_unpack_loss_mask_preserves_compact_semantics():
    assert unpack_loss_mask("2,3,1", expected_length=6) == [0, 0, 1, 1, 1, 0]


def test_prepare_data_preserves_order_tokens_and_supervision(tmp_path):
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "data_id": "a",
                    "input_ids": [1, 2, 3],
                    "packed_loss_mask": "1,2",
                    "loss_tokens": 2,
                },
                {
                    "data_id": "b",
                    "input_ids": [4, 5],
                    "packed_loss_mask": "0,1,1",
                    "loss_tokens": 1,
                },
            )
        )
    )
    output = tmp_path / "specforge.jsonl"

    manifest = prepare_data(source, output)

    assert manifest["rows"] == 2
    assert manifest["input_tokens"] == 5
    assert manifest["supervised_positions"] == 3
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [1, 0]},
    ]


def test_prepare_initial_only_renames_equivalent_backbone_keys(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source"
    source.mkdir()
    state = {
        "context_proj.weight": torch.arange(6).reshape(2, 3),
        "context_norm.weight": torch.ones(2),
        "final_norm.weight": torch.full((2,), 2),
        "layers.0.weight": torch.full((2, 2), 3),
    }
    safetensors.save_file(state, source / "model.safetensors")
    config = tmp_path / "config.json"
    config.write_text('{"architectures":["DSparkDraftModel"]}')

    manifest = prepare_initial(source, config, tmp_path / "adapted")
    adapted = safetensors.load_file(tmp_path / "adapted" / "model.safetensors")

    assert manifest["tensor_count"] == 4
    assert set(adapted) == {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "layers.0.weight",
    }
    assert torch.equal(adapted["fc.weight"], state["context_proj.weight"])


@pytest.mark.parametrize("packed", ["", "1,nope", "1,-1"])
def test_unpack_loss_mask_rejects_invalid_input(packed):
    with pytest.raises(ValueError):
        unpack_loss_mask(packed, expected_length=1)


def test_summarize_metrics_uses_slowest_synchronized_rank():
    metrics = {
        1: [
            {"perf/optimizer_step_time_s": 2.0, "perf/global_samples_per_second": 2.0},
            {"perf/optimizer_step_time_s": 2.1, "perf/global_samples_per_second": 4 / 2.1},
        ],
        2: [
            {"perf/optimizer_step_time_s": 1.0, "perf/global_samples_per_second": 4.0},
            {"perf/optimizer_step_time_s": 1.1, "perf/global_samples_per_second": 4 / 1.1},
        ],
    }

    result = summarize_metrics(
        metrics,
        expected_steps=2,
        expected_ranks=2,
        expected_documents=8,
        expected_input_tokens=80,
        expected_supervised_positions=40,
        warmup_steps=1,
    )

    assert result["full"]["seconds"] == pytest.approx(3.2)
    assert result["steady"]["documents"] == 4
    assert result["steady"]["estimated_input_tokens_per_second"] == pytest.approx(40 / 1.1)
    assert "full-corpus mean" in result["steady"]["token_rate_estimation"]


def test_summarize_metrics_rejects_missing_rank():
    with pytest.raises(ValueError, match="rank metrics"):
        summarize_metrics(
            {1: [{"perf/optimizer_step_time_s": 1, "perf/global_samples_per_second": 1}]},
            expected_steps=1,
            expected_ranks=2,
            expected_documents=1,
            expected_input_tokens=1,
            expected_supervised_positions=1,
            warmup_steps=0,
        )


def _wall_rank(directory, rank, start, completions, tokens, ids=None, host="node"):
    rows = [{"event": "loop_start", "clock": "perf_counter_ns", "hostname": host,
             "rank": rank, "clock_ns": start}]
    rows.extend({"event": "optimizer_complete", "rank": rank, "step": i + 1,
                 "clock_ns": stamp, "input_tokens": count,
                 "sample_ids": [ids[i] if ids else f"{rank}-{i}"]}
                for i, (stamp, count) in enumerate(zip(completions, tokens)))
    (directory / f"rank-{rank}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_wall_window_uses_actual_tokens_and_continuous_clock(tmp_path):
    _wall_rank(tmp_path, 0, 1_000_000_000, [3_000_000_000, 6_000_000_000], [2, 19])
    _wall_rank(tmp_path, 1, 1_100_000_000, [3_100_000_000, 6_200_000_000], [3, 76])
    result = benchmark.summarize_wall_observations(
        tmp_path, expected_steps=2, expected_ranks=2, expected_documents=4,
        expected_input_tokens=100, warmup_steps=1,
    )
    assert result["full"]["seconds"] == pytest.approx(5.2)
    assert result["full"]["input_tokens_per_second"] == pytest.approx(100 / 5.2)
    assert result["steady"]["input_tokens"] == 95  # not the corpus-mean estimate of 50
    assert result["steady"]["input_tokens_per_second"] == pytest.approx(95 / 3.1)


@pytest.mark.parametrize("fault", ["missing_rank", "gap", "repeated_id", "different_host", "clock", "tokens"])
def test_wall_observer_rejects_incomplete_or_invalid_evidence(tmp_path, fault):
    _wall_rank(tmp_path, 0, 1, [3, 6], [2, 19], ids=["same", "same"] if fault == "repeated_id" else None)
    if fault != "missing_rank":
        _wall_rank(tmp_path, 1, 1, [3, 2] if fault == "clock" else [3, 6], [3, 76],
                   host="other" if fault == "different_host" else "node")
    if fault == "gap":
        p = tmp_path / "rank-0.jsonl"
        p.write_text(p.read_text().replace('"step": 2', '"step": 3'))
    with pytest.raises(ValueError):
        benchmark.summarize_wall_observations(
            tmp_path, expected_steps=2, expected_ranks=2, expected_documents=4,
            expected_input_tokens=101 if fault == "tokens" else 100, warmup_steps=1,
        )


def test_observer_delegates_batches_and_ignores_prefetched_work(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.performance.specforge_wall_observer import patch_controller, patch_loader

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("ASPEC_SPECFORGE_OBSERVATION_DIR", str(tmp_path))
    tensor = object()

    class Loader:
        def _make_batch(self, refs):
            return SimpleNamespace(metadata={}, sample_ids=[r.sample_id for r in refs],
                                   tensors={"input_ids": tensor})

        def __iter__(self):
            yield self.used

    patch_loader(Loader)
    data = Loader()
    data.used = data._make_batch([SimpleNamespace(num_tokens=7, sample_id="trained")])
    data.unused = data._make_batch([SimpleNamespace(num_tokens=50, sample_id="prefetched")])
    logged = []

    class Controller:
        log_interval = 1

        def __init__(self):
            self.logger = lambda metrics, step: logged.append((metrics, step))

        def _fit(self, stream, progress):
            for batch in stream:
                assert batch is data.used and batch.tensors["input_ids"] is tensor
                self.logger({"perf/optimizer_step_time_s": 1.0}, 1)
            self.logger({"eval/loss": 0.0}, 1)
            return 1

    patch_controller(Controller)
    controller = Controller()
    original_logger = controller.logger
    assert controller._fit(data, None) == 1
    assert controller.logger is original_logger
    rows = [json.loads(line) for line in (tmp_path / "rank-0.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["sample_ids"] == ["trained"]
    assert rows[1]["input_tokens"] == 7
    assert len(logged) == 2
