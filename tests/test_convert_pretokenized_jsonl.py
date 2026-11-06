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
from pathlib import Path

import pytest

from aspec.data.dataset import load_conversation_dataset
from tools.convert_pretokenized_jsonl import convert


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_conversion_produces_typed_loader_boundary(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "legacy.jsonl"
    destination = tmp_path / "canonical.parquet"
    _write_jsonl(
        source,
        [
            {"id": "a", "input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 0]},
            {
                "data_id": "b",
                "input_ids": [5, 6, 7],
                "packed_loss_mask": "1,2",
                "synthetic": True,
            },
        ],
    )

    manifest = convert(source, destination, batch_rows=1)

    assert manifest["rows"] == 2
    assert manifest["schema_version"] == "aspec.pretokenized-conversion/v1"
    assert destination.with_suffix(".parquet.manifest.json").is_file()

    args = type(
        "Args",
        (),
        {
            "train_data_path": str(destination),
            "max_seq_length": 4,
            "defer_tokenization": False,
            "chat_template": None,
            "renderer": None,
        },
    )()
    loaded = load_conversation_dataset(args)
    assert [row["data_id"] for row in loaded] == ["a", "b"]
    assert loaded[0]["packed_loss_mask"] == "1,2,1"
    assert loaded[1]["metadata"]["source_metadata_json"] == '{"synthetic":true}'


@pytest.mark.parametrize(
    "row, message",
    [
        ({"id": "a", "input_ids": [], "loss_mask": []}, "input_ids"),
        ({"id": "a", "input_ids": [1, 2], "loss_mask": [1]}, "length"),
        ({"id": "a", "input_ids": [1, 2], "loss_mask": [0, 0]}, "no supervised"),
        (
            {
                "id": "a",
                "input_ids": [1, 2],
                "loss_mask": [0, 1],
                "packed_loss_mask": "2",
            },
            "disagrees",
        ),
    ],
)
def test_conversion_rejects_invalid_contract(tmp_path, row, message):
    pytest.importorskip("pyarrow")
    source = tmp_path / "legacy.jsonl"
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match=message):
        convert(source, tmp_path / "canonical.parquet")


def test_conversion_rejects_duplicate_identity(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "legacy.jsonl"
    row = {"id": "same", "input_ids": [1, 2], "loss_mask": [0, 1]}
    _write_jsonl(source, [row, row])

    with pytest.raises(ValueError, match="duplicate data_id"):
        convert(source, tmp_path / "canonical.parquet", batch_rows=1)
