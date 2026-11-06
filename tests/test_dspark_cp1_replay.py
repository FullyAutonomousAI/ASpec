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

from types import SimpleNamespace

import pytest
import torch

from tools.validation.dspark_cp1_replay import transplant_plain


def test_transplant_changes_only_the_exact_actual_state_not_lr_or_adam_phase():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    master = model.weight.detach().float().clone()
    state = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.zeros_like(master),
        "exp_avg_sq": torch.zeros_like(master),
    }
    optimizer = SimpleNamespace(
        model_params=[model.weight],
        fp32_params=[master],
        optimizer=SimpleNamespace(state={master: state}, param_groups=[{"lr": 0.0003}]),
    )
    source = {
        "weight": {
            "model": torch.full_like(model.weight, 2),
            "master": torch.full_like(master, 2.001),
            "exp_avg": torch.full_like(master, 0.1),
            "exp_avg_sq": torch.full_like(master, 0.2),
            "optimizer_step": 2,
        }
    }
    transplant_plain(model, optimizer, source)
    assert torch.equal(model.weight, source["weight"]["model"])
    assert torch.equal(master, source["weight"]["master"])
    assert torch.equal(state["exp_avg"], source["weight"]["exp_avg"])
    assert torch.equal(state["exp_avg_sq"], source["weight"]["exp_avg_sq"])
    assert state["step"].item() == 2
    assert optimizer.optimizer.param_groups[0]["lr"] == 0.0003
    source["weight"]["optimizer_step"] = 3
    with pytest.raises(ValueError, match="Adam step phase"):
        transplant_plain(model, optimizer, source)
