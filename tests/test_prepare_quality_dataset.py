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

import pytest

from tools import prepare_quality_dataset


class _Dataset:
    _fingerprint = "test-source-fingerprint"

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def _conversation(index):
    return {
        "conversations": [
            {"from": "human", "value": f"question {index}"},
            {"from": "gpt", "value": f"answer {index} " + "x" * 50},
        ]
    }


def test_prepare_split_is_deterministic_disjoint_and_hashed(tmp_path, monkeypatch):
    dataset = _Dataset([_conversation(index) for index in range(8)])
    monkeypatch.setattr(
        prepare_quality_dataset,
        "load_dataset",
        lambda _: {"train": dataset},
    )

    manifest = prepare_quality_dataset.prepare_split(
        output_dir=tmp_path,
        train_size=3,
        eval_size=2,
        seed=17,
    )

    train_rows = [json.loads(line) for line in open(manifest["train"]["path"])]
    eval_rows = [json.loads(line) for line in open(manifest["eval"]["path"])]
    train_ids = {row["id"] for row in train_rows}
    eval_ids = {row["id"] for row in eval_rows}
    assert len(train_ids) == 3
    assert len(eval_ids) == 2
    assert train_ids.isdisjoint(eval_ids)
    assert manifest["source_fingerprint"] == dataset._fingerprint
    assert manifest["train"]["sha256"]
    assert manifest["eval"]["sha256"]
    assert all(message["role"] in {"user", "assistant"} for row in train_rows for message in row["conversations"])


def test_prepare_split_refuses_to_overwrite_artifacts(tmp_path, monkeypatch):
    dataset = _Dataset([_conversation(index) for index in range(4)])
    monkeypatch.setattr(
        prepare_quality_dataset,
        "load_dataset",
        lambda _: {"train": dataset},
    )
    kwargs = {"output_dir": tmp_path, "train_size": 1, "eval_size": 1, "seed": 3}
    prepare_quality_dataset.prepare_split(**kwargs)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_quality_dataset.prepare_split(**kwargs)
