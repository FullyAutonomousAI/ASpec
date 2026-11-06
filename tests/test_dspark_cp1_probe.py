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

import os

import pytest
import torch

from tools.validation.dspark_cp1_probe import gradient_difference, rows_for_mode, to_device


def test_probe_compares_all_coordinates_not_only_gradient_norm():
    reference = {"p": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)}
    same = gradient_difference(reference, reference)
    assert same["relative_l2_diff"] == 0
    assert same["unequal"] == 0
    actual = {"p": torch.tensor([-1.0, 2.0, 3.0], dtype=torch.bfloat16)}
    result = gradient_difference(reference, actual)
    assert result["square_ref"] == result["square_actual"] == 14
    assert result["square_diff"] == 4
    assert result["unequal"] == 1
    assert result["relative_l2_diff"] == pytest.approx((4 / 14) ** 0.5)
    with pytest.raises(ValueError, match="parameter sets"):
        gradient_difference(reference, {"q": actual["p"]})


def test_probe_preserves_packed_input_structure_and_precision():
    packed = [
        {
            "input_ids": torch.ones(1, 7, dtype=torch.long),
            "hidden_states_list": [torch.ones(1, 7, 3, dtype=torch.bfloat16)],
            "anchor_seed": None,
        }
    ]
    assert rows_for_mode(packed, False) is packed
    result = to_device(packed[0], "cpu")
    assert result["hidden_states_list"][0].dtype == torch.bfloat16
    assert result["anchor_seed"] is None


@pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="opt-in CUDA scalar coefficient regression",
)
def test_native_normalization_keeps_fp32_tensor_coefficient():
    from types import SimpleNamespace

    from aspec.training.dflash_trainer import DFlashTrainer

    values = (torch.arange(-4000, 4000, device="cuda").float() / 100).bfloat16()
    parameter = torch.nn.Parameter(torch.zeros_like(values))
    parameter.grad = values.clone()
    denominator = torch.tensor(951.645751953125, device="cuda", dtype=torch.float32)
    trainer = SimpleNamespace(optimizer=SimpleNamespace(model_params=[parameter]), dp_group=None)
    expected = values * (2 / denominator)
    rounded_python_control = values * (2 / denominator.item())
    DFlashTrainer._normalize_accumulated_gradients(trainer, denominator, 2)
    assert torch.equal(parameter.grad, expected)
    assert not torch.equal(parameter.grad, rounded_python_control)
