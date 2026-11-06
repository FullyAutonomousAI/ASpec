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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tools.prepare_selected_answers import original_answers, prepare, selected_sources


def test_join_uses_source_identity_original_answer_and_selection_order(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "id": key,
            "conversations": [
                {"role": "user", "content": f"question {key}"},
                {"role": "assistant", "content": f"original answer {key}"},
                {"role": "user", "content": "later turn"},
            ],
        }
        for key in ("a", "b")
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = [("frozen-b", "b"), ("frozen-a", "a")]
    result = original_answers(source, selected)
    assert [row[0] for row in result] == ["frozen-b", "frozen-a"]
    assert result[0][2] == rows[1]["conversations"][:2]
    assert result[1][2][-1]["content"] == "original answer a"
    with pytest.raises(ValueError, match="missing"):
        original_answers(source, [("c", "missing")])
    source.write_text(source.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        original_answers(source, selected)


@pytest.mark.parametrize("metadata", [{}, {"source_id": ""}, {"source_id": 3}])
def test_selection_rejects_lost_source_identity(tmp_path, metadata):
    path = tmp_path / "selection.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"data_id": "x", "source_metadata_json": json.dumps(metadata)}]), path
    )
    with pytest.raises(ValueError, match="source_id"):
        selected_sources(path)


def test_selection_preserves_order_and_rejects_duplicates(tmp_path):
    path = tmp_path / "selection.parquet"
    rows = [
        {"data_id": key, "source_metadata_json": json.dumps({"source_id": f"raw-{key}"})}
        for key in ("b", "a")
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert selected_sources(path) == [("b", "raw-b"), ("a", "raw-a")]
    pq.write_table(pa.Table.from_pylist(rows + rows[:1]), path)
    with pytest.raises(ValueError, match="duplicate"):
        selected_sources(path)


@pytest.mark.parametrize("identity", [None, ""])
def test_selection_rejects_lost_document_identity(tmp_path, identity):
    path = tmp_path / "selection.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"data_id": identity, "source_metadata_json": json.dumps({"source_id": "raw-a"})}]
        ),
        path,
    )
    with pytest.raises(ValueError, match="data_id"):
        selected_sources(path)


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_prepare_retokens_answers_and_filters_without_truncating(tmp_path, monkeypatch, trust_remote_code):
    from transformers import AutoTokenizer

    from aspec.data import parse

    selection = tmp_path / "selection.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "data_id": f"frozen-{key}",
                    "source_metadata_json": json.dumps({"source_id": key}),
                    # Existing tokens must not reach the Ling parser.
                    "input_ids": [99999],
                }
                for key in ("keep", "long", "short")
            ]
        ),
        selection,
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "id": key,
                    "conversations": [
                        {"role": "user", "content": f"question {key}"},
                        {"role": "assistant", "content": key},
                    ],
                }
            )
            + "\n"
            for key in ("short", "long", "keep")
        )
    )
    target = tmp_path / "ling"
    target.mkdir()
    (target / "tokenizer_config.json").write_text('{"chat_template":"test"}')
    tokenizer = object()
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: (
            tokenizer if path == target and kwargs == {"trust_remote_code": trust_remote_code} else None
        ),
    )
    parsed = {
        "keep": (torch.arange(8), torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])),
        "long": (torch.arange(9), torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])),
        "short": (torch.arange(4), torch.tensor([0, 0, 0, 1])),
    }

    class Parser:
        def format(self, messages, **kwargs):
            assert kwargs == {"add_generation_prompt": False, "enable_thinking": False}
            assert messages[0]["role"] == "user"
            return messages[1]["content"]

        def parse(self, formatted, max_length, **kwargs):
            assert max_length == 9
            assert kwargs == {"preformatted": True, "last_turn_only": False}
            return parsed[formatted]

    def create_parser(actual, template):
        assert actual is tokenizer
        assert template is not None
        return Parser()

    monkeypatch.setattr(parse, "create_parser", create_parser)
    output = tmp_path / "prepared"
    manifest = prepare(
        selection=selection,
        source=source,
        target=target,
        template="ling3",
        output=output,
        max_length=8,
        min_loss_tokens=2,
        **({"trust_remote_code": True} if trust_remote_code else {}),
    )
    assert manifest["trust_remote_code"] is trust_remote_code
    assert manifest["counts"] == {
        "selected": 3,
        "over_length": 1,
        "short_supervision": 1,
        "admitted": 1,
    }
    assert not manifest["answer_regeneration"] and not manifest["quality_claim"]
    assert manifest["input_tokens"] == 8 and manifest["supervised_positions"] == 2
    rows = pq.read_table(output / "documents.parquet").to_pylist()
    assert len(rows) == 1
    assert rows[0]["data_id"] == "frozen-keep"
    assert rows[0]["input_ids"] == list(range(8))
    assert rows[0]["packed_loss_mask"] == "6,2"
    assert json.loads(rows[0]["source_metadata_json"]) == {"source_id": "keep"}
