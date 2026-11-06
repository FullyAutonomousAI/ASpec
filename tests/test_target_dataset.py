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

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tools.prepare_target_dataset import collect_resumed, regenerate, select_prompts


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def conversation(question, answer="source answer MUST NOT enter generation"):
    return {"conversations": [{"role": "user", "content": question},
                               {"role": "assistant", "content": answer}]}


def test_selection_excludes_source_answers_and_deduplicates_questions(tmp_path):
    train, heldout = tmp_path / "train", tmp_path / "eval"
    write_rows(train, [conversation("shared question"), conversation("train question"),
                       conversation("train  question")])
    write_rows(heldout, [conversation("shared question")])
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2]
    selected, stats = select_prompts({"train": train, "eval": heldout}, tokenizer,
                                     max_length=10, min_answer_tokens=2)
    assert len(selected["train"]) == len(selected["eval"]) == 1
    assert selected["train"][0]["id"] != selected["eval"][0]["id"]
    assert stats["train"]["duplicate_questions"] == 2
    for call in tokenizer.apply_chat_template.call_args_list:
        assert all(m["role"] != "assistant" for m in call.args[0])
        assert call.kwargs["enable_thinking"] is False


def test_generation_keeps_exact_ids_mask_and_resumes_without_duplicate_work(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"id": "a" * 64, "source_id": "raw_1", "input_ids": [1, 2]}]
    engine = Mock()
    engine.async_generate = AsyncMock(return_value={"output_ids": [7, 8, 9],
                                     "meta_info": {"finish_reason": {"type": "stop"}}})
    asyncio.run(regenerate(engine, rows, path, batch_size=4, max_length=8, seed=42))
    row = json.loads(path.read_text())
    assert row["input_ids"] == [1, 2, 7, 8, 9]
    assert row["prompt_input_ids"] == [1, 2]
    assert row["loss_mask"] == [0, 0, 1, 1, 1]
    assert engine.async_generate.call_args.kwargs["sampling_params"]["max_new_tokens"] == 6
    asyncio.run(regenerate(engine, rows, path, batch_size=4, max_length=8, seed=42))
    assert engine.async_generate.call_count == 1
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("corrupt", ["duplicate", "prompt", "mask", "partial"])
def test_resume_rejects_corrupt_or_changed_journal(tmp_path, corrupt):
    path = tmp_path / "train.jsonl"
    row = {"id": "a", "prompt_input_ids": [1], "input_ids": [1, 3], "loss_mask": [0, 1]}
    if corrupt == "prompt":
        row["prompt_input_ids"] = [2]
    if corrupt == "mask":
        row["loss_mask"] = [1, 1]
    write_rows(path, [row, row] if corrupt == "duplicate" else [row])
    if corrupt == "partial":
        with path.open("a") as stream:
            stream.write('{"id":')
    with pytest.raises(ValueError):
        collect_resumed(path, [{"id": "a", "input_ids": [1]}])


def test_generation_failure_does_not_publish_a_fake_answer(tmp_path):
    engine = Mock()
    engine.async_generate = AsyncMock(return_value={"output_ids": [3],
                                     "meta_info": {"finish_reason": {"type": "abort"}}})
    path = tmp_path / "out"
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(regenerate(engine, [{"id": "a" * 64, "source_id": "raw", "input_ids": [1]}],
                               path, batch_size=1, max_length=8, seed=0))
    assert not path.read_text()


def test_completed_short_request_refills_without_waiting_for_a_long_peer(tmp_path):
    async def run():
        blocked, third_started = asyncio.Event(), asyncio.Event()

        async def generate(*, input_ids, sampling_params):
            if input_ids == [1]:
                await third_started.wait()
                blocked.set()
            if input_ids == [3]:
                third_started.set()
            return {"output_ids": [9], "meta_info": {"finish_reason": {"type": "stop"}}}

        rows = [{"id": f"{i:064x}", "source_id": str(i), "input_ids": [i]} for i in (1, 2, 3)]
        await asyncio.wait_for(regenerate(Mock(async_generate=generate), rows, tmp_path / "out",
                                          batch_size=2, max_length=8, seed=0), timeout=1)
        assert blocked.is_set()
        assert collect_resumed(tmp_path / "out", rows) == {row["id"] for row in rows}

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["weights", "source", "seed"])
def test_resume_rejects_changed_generation_identity_before_starting_an_engine(tmp_path, monkeypatch, changed):
    from tools import prepare_target_dataset as tool

    target = tmp_path / "target"
    target.mkdir()
    weights = target / "model.safetensors"
    weights.write_bytes(b"original weights")
    train, heldout = tmp_path / "train", tmp_path / "eval"
    train.write_text("original train")
    heldout.write_text("original eval")
    argv = ["prepare", "--target", str(target), "--train-data", str(train),
            "--eval-data", str(heldout), "--output-dir", str(tmp_path / "output")]
    monkeypatch.setattr(sys, "argv", argv)
    real_version = tool.importlib.metadata.version
    monkeypatch.setattr(tool.importlib.metadata, "version",
                        lambda name: "0.5.21" if name == "sglang" else real_version(name))
    real_write = tool.write_json

    class ManifestCreated(Exception):
        pass

    def stop_after_manifest(path, value):
        real_write(path, value)
        raise ManifestCreated

    # No tokenizer model download or SGLang/GPU process is needed to exercise
    # the actual CLI's binding and fail-closed resume gate.
    monkeypatch.setattr(tool, "write_json", stop_after_manifest)
    with pytest.raises(ManifestCreated):
        tool.main()
    monkeypatch.setattr(tool, "write_json", real_write)
    if changed == "weights":
        weights.write_bytes(b"different weights")
    elif changed == "source":
        train.write_text("different train")
    else:
        argv = [*argv, "--seed", "123"]
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises(ValueError, match="identity changed"):
        tool.main()


def test_new_generation_cannot_overwrite_a_nonempty_directory(tmp_path, monkeypatch):
    from tools import prepare_target_dataset as tool

    (tmp_path / "unrelated.txt").write_text("keep me")
    monkeypatch.setattr(sys, "argv", ["prepare", "--target", str(tmp_path),
                                    "--train-data", "train", "--eval-data", "eval",
                                    "--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="nonempty"):
        tool.main()
    assert (tmp_path / "unrelated.txt").read_text() == "keep me"


def test_generated_journal_to_typed_training_boundary_and_decode_prefix(tmp_path):
    from aspec.data.dataset import load_conversation_dataset
    from tools.convert_pretokenized_jsonl import convert
    from tools.evaluate_draft import load_prompts

    journal, parquet = tmp_path / "generated.jsonl", tmp_path / "generated.parquet"
    output_ids = list(range(3, 20))
    engine = Mock(async_generate=AsyncMock(return_value={
        "output_ids": output_ids, "meta_info": {"finish_reason": {"type": "stop"}},
    }))
    rows = [{"id": "a" * 64, "source_id": "raw", "input_ids": [1, 2]}]
    asyncio.run(regenerate(engine, rows, journal, batch_size=2, max_length=32, seed=0))
    convert(journal, parquet, batch_rows=256)
    args = SimpleNamespace(train_data_path=str(parquet), max_seq_length=32,
                           min_loss_tokens=14, defer_tokenization=False,
                           cache_dir=str(tmp_path / "cache"), prompt_key="conversations",
                           target_model_path="must-not-be-rendered", chat_template=None)
    loaded = load_conversation_dataset(args)
    assert len(loaded) == 1
    assert loaded[0]["data_id"] == rows[0]["id"]
    assert loaded[0]["input_ids"].tolist() == [1, 2, *output_ids]
    assert loaded[0]["packed_loss_mask"] == "2,17"
    assert load_prompts(journal, Mock(), 1) == [[1, 2]]
