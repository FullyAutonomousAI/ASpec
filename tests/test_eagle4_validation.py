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

"""Validation CLIs reject invalid budgets before optional references or CUDA."""
import importlib

import pytest
import torch


@pytest.mark.parametrize("module", ["eagle4_upstream", "eagle4_candidate_state"])
@pytest.mark.parametrize("steps", [0, -1])
def test_eagle4_validation_rejects_nonpositive_steps_before_cuda(monkeypatch, tmp_path, module, steps):
    entry = importlib.import_module(f"tools.validation.{module}")
    monkeypatch.setattr("sys.argv", [module, "--output", str(tmp_path / "result.json"), "--steps", str(steps)])
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CUDA initialized for invalid input"))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


@pytest.mark.parametrize("module", ["eagle4_upstream", "eagle4_candidate_state"])
def test_eagle4_validation_help_does_not_require_cuda_or_upstream(monkeypatch, module):
    entry = importlib.import_module(f"tools.validation.{module}")
    monkeypatch.setattr("sys.argv", [module, "--help"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CUDA initialized for help"))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 0
