# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_data_utils.py. Namespace bindings and ASpec regression coverage adapted.

import os
from pathlib import Path

import pytest

from aspec.data import utils as data_utils
from aspec.data.utils import is_local_data_path

# ---------------------------------------------------------------------------
# is_local_data_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "./data.jsonl",
        "../data.jsonl",
        "/absolute/data.jsonl",
        "~/data.jsonl",
        "train.json",
        "train.parquet",
        "data.csv",
    ],
)
def test_is_local_data_path_true(path):
    assert is_local_data_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "HuggingFaceH4/ultrachat_200k",
        "allenai/tulu-v2-sft-mixture",
        "namespace/dataset-name",
    ],
)
def test_is_local_data_path_false_for_hub_ids(path):
    assert is_local_data_path(path) is False


def test_is_local_data_path_existing_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_dataset").mkdir()
    assert is_local_data_path("my_dataset") is True


# ---------------------------------------------------------------------------
# load_hf_dataset
# ---------------------------------------------------------------------------


def test_load_hf_dataset_treats_existing_relative_dir_as_local(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "my_dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "sample.jsonl").write_text('{"x": 1}\n')

    rel_path = Path(os.path.relpath(data_dir, Path.cwd()))

    called = {}

    def fake_load_dataset(fmt, data_files, split, streaming):
        called["fmt"] = fmt
        called["data_files"] = data_files
        called["split"] = split
        called["streaming"] = streaming
        return "LOCAL_DATASET"

    monkeypatch.setattr(data_utils, "load_dataset", fake_load_dataset)

    out = data_utils.load_hf_dataset(str(rel_path))

    assert out == "LOCAL_DATASET"
    assert called["fmt"] == "json"
    assert called["split"] == "train"
    assert called["streaming"] is True
    assert any(str(f).endswith("sample.jsonl") for f in called["data_files"])
