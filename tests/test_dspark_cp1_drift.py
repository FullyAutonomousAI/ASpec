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

import pytest
import torch

from tools.validation.dspark_cp1_drift import CapturingReference, owned_inputs


def test_capture_owns_inputs_and_does_not_copy_or_mutate_the_head():
    ids = torch.arange(7).view(1, -1)
    feature = torch.ones(1, 7, 3, dtype=torch.bfloat16)
    head = torch.randn(11, 3)
    result = owned_inputs(
        {
            "input_ids": ids,
            "hidden_states_list": [feature],
            "lm_head_weight": head,
            "anchor_seed": None,
        }
    )
    assert "lm_head_weight" not in result
    assert result["hidden_states_list"][0].dtype == torch.bfloat16
    ids.zero_()
    feature.zero_()
    assert result["input_ids"].tolist() == [list(range(7))]
    assert result["hidden_states_list"][0].sum() == 21
    with pytest.raises(TypeError, match="Unsupported forward input"):
        owned_inputs({"unexpected": object()})


def test_capture_updates_reference_step_before_the_native_audit(monkeypatch):
    from types import SimpleNamespace

    reference = CapturingReference.__new__(CapturingReference)
    reference.trainer = SimpleNamespace(global_step=7)
    reference.audit_trainer = SimpleNamespace(global_step=0)
    reference.microbatch = 2
    seen = []
    monkeypatch.setattr(
        "tools.validation.dspark_cp1_drift.alignment.TorchReference.before_update",
        lambda self: seen.append(self.audit_trainer.global_step),
    )
    reference.before_update()
    assert seen == [7]
    assert reference.microbatch == 0
