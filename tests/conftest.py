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

"""Pytest collection policy for the fast default unit-test suite."""

import pytest

# This is a manually launched diagnostic program, not a pytest test module.
# Ignoring it avoids network probing and heavyweight backend imports during
# ordinary test collection while preserving ``python tests/...`` usage.
collect_ignore = ["test_sglang_engine_integration.py"]


@pytest.fixture(autouse=True)
def release_recipe_inputs(monkeypatch):
    """Planning fixtures only: never used by the public launcher or GPU training."""
    monkeypatch.setenv("ASPEC_TRAIN_DATA", "/tmp/aspec-test/train.parquet")
    monkeypatch.setenv("ASPEC_EVAL_DATA", "/tmp/aspec-test/eval.parquet")
    monkeypatch.setenv("ASPEC_INITIAL_DRAFT", "/tmp/aspec-test/eagle4-asset")

    from pathlib import Path
    monkeypatch.setenv("ASPEC_DRAFT_CONFIG", str(Path(__file__).parents[1] / "aspec/config/dspark_draft_config.json"))
    monkeypatch.setenv("ASPEC_OUTPUT_DIR", "/tmp/aspec-test/output")
    monkeypatch.setenv("ASPEC_CACHE_DIR", "/tmp/aspec-test/cache")
