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

from tools.validation.dspark_teacher_contract import (
    fixture_teacher_contract,
    teacher_contract,
    teacher_last_hidden_states,
)


@pytest.mark.parametrize("engine", ["sgl", "sglang", "hf"])
def test_postnorm_preserves_the_cached_tensor_without_an_extra_norm(engine):
    value = torch.arange(12).view(1, 3, 4).bfloat16()
    head = SimpleNamespace(norm=None, _aspec_teacher_contract=teacher_contract(engine))
    assert teacher_last_hidden_states(head, value) is value
    assert head._aspec_teacher_contract["additional_final_norm_calls"] == 0


@pytest.mark.parametrize("engine,override", [("vllm", None), ("offline", True), ("hf", True)])
def test_prenorm_calls_native_norm_exactly_once_without_grad(engine, override):
    calls = []

    def norm(value):
        calls.append(torch.is_grad_enabled())
        return value + 3

    head = SimpleNamespace(norm=norm, _aspec_teacher_contract=teacher_contract(engine, override))
    value = torch.ones(1, 3, 4, requires_grad=True)
    actual = teacher_last_hidden_states(head, value)
    assert torch.equal(actual, value + 3)
    assert not actual.requires_grad
    assert calls == [False]


@pytest.mark.parametrize("engine", ["vllm", "offline"])
def test_explicit_postnorm_override_bypasses_norm(engine):
    value = torch.ones(1)
    head = SimpleNamespace(norm=None, _aspec_teacher_contract=teacher_contract(engine, False))
    assert teacher_last_hidden_states(head, value) is value


@pytest.mark.parametrize("engine,override", [("unknown", None), (None, None), ("offline", None),
                                           ("hf", "false"), ("hf", 0)])
def test_unknown_or_ambiguous_feature_contract_fails_closed(engine, override):
    with pytest.raises(ValueError):
        teacher_contract(engine, override)


def test_norm_module_and_recorded_contract_must_agree():
    corrupted = dict(teacher_contract("sglang"), additional_final_norm_calls=1)
    for head in (SimpleNamespace(norm=None),
                 SimpleNamespace(norm=None, _aspec_teacher_contract=teacher_contract("vllm")),
                 SimpleNamespace(norm=lambda x: x, _aspec_teacher_contract=teacher_contract("sglang")),
                 SimpleNamespace(norm=None, _aspec_teacher_contract=corrupted)):
        with pytest.raises(ValueError, match="contract"):
            teacher_last_hidden_states(head, torch.ones(1))


def test_fixture_requires_an_explicit_offline_normalization_stage(tmp_path):
    config = tmp_path / "fast.yaml"
    config.write_text("inference:\n  inference_engine_type: offline\n")
    with pytest.raises(ValueError, match="explicit"):
        fixture_teacher_contract(tmp_path)
    config.write_text("inference:\n  inference_engine_type: offline\n  last_hidden_states_prenorm: false\n")
    assert fixture_teacher_contract(tmp_path) == teacher_contract("offline", False)
