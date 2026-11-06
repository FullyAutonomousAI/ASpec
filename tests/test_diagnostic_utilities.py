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


import logging
from unittest.mock import patch

import pytest
import torch

from aspec.utils.logging import _get_logger_level
from aspec.utils.misc import get_free_port
from aspec.utils.train_dump import compare_eagle3_batches, compare_eagle3_outputs


@pytest.mark.parametrize("name", ["not_a_level", "getLogger"])
def test_bad_log_level_has_safe_fallback(monkeypatch, name):
    monkeypatch.setenv("ASPEC_LOG_LEVEL", name)
    assert _get_logger_level() == logging.WARNING


def test_port_search_stops_at_end_of_range():
    # Stop the old unbounded search deterministically without hanging the test.
    def available(port):
        assert port <= 65535, "port search crossed the TCP port range"
        return False

    with patch("aspec.utils.misc._is_port_available", side_effect=available):
        with pytest.raises(RuntimeError, match="port"):
            get_free_port(65535)


@pytest.mark.parametrize("case", ["empty", "missing_file", "missing_tensor", "double", "target"])
def test_dump_comparison_requires_actual_matching_evidence(tmp_path, case):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    if case == "empty":
        result = compare_eagle3_batches(str(a), str(b), verbose=False)
    elif case == "missing_file":
        torch.save({"total_loss": torch.tensor(1.0)}, a / "step.pt")
        result = compare_eagle3_batches(str(a), str(b), verbose=False)
    else:
        left = {"total_loss": torch.tensor(1.0)}
        right = dict(left)
        if case == "missing_tensor":
            left["gradients"] = {"weight": torch.ones(2)}
        elif case == "double":
            left["total_loss"] = torch.tensor(1.0, dtype=torch.float64)
            right["total_loss"] = torch.tensor(1.0 + 1e-10, dtype=torch.float64)
        else:
            left["target"] = torch.tensor([1.0, 2.0])
            right["target"] = torch.tensor([1.0, 3.0])
        torch.save(left, a / "step.pt")
        torch.save(right, b / "step.pt")
        result = compare_eagle3_outputs(str(a / "step.pt"), str(b / "step.pt"), rtol=0, atol=0)
    assert not result["match"], result


@pytest.mark.parametrize(
    "start,count", [(0, 1), (65536, 1), (10, 0), (65535, 2), (True, 1), (10, True)]
)
def test_invalid_port_range_is_rejected(start, count):
    with pytest.raises(ValueError, match="port"):
        get_free_port(start, count)


def test_port_search_includes_last_valid_candidate():
    with patch("aspec.utils.misc._is_port_available", side_effect=lambda port: port >= 65534):
        assert get_free_port(65532, consecutive=2) == 65534


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool, torch.complex128],
)
def test_identical_dump_has_positive_tensor_evidence(tmp_path, dtype):
    p = tmp_path / "step.pt"
    torch.save(
        {"total_loss": torch.ones(2, dtype=dtype), "gradients": {"w": torch.empty(0, dtype=dtype)}},
        p,
    )
    result = compare_eagle3_outputs(str(p), str(p), rtol=0, atol=0)
    assert result["match"]
    assert result["summary"]["tensors_compared"] == 2


def test_empty_dumps_are_not_precision_evidence(tmp_path):
    p = tmp_path / "step.pt"
    torch.save({"step": 1}, p)
    assert not compare_eagle3_outputs(str(p), str(p))["match"]
