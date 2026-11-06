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

from tools.performance.specforge_draft_benchmark import _adapt_feature, _finite_float


def test_adapt_feature_renames_last_hidden_and_squeezes_token_axes():
    torch = pytest.importorskip("torch")
    raw = {
        "input_ids": torch.arange(3)[None],
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "hidden_states": torch.zeros(1, 3, 4),
        "last_hidden_states": torch.ones(1, 3, 2),
    }

    adapted = _adapt_feature(raw)

    assert tuple(adapted["input_ids"].shape) == (3,)
    assert tuple(adapted["loss_mask"].shape) == (3,)
    assert adapted["target_last_hidden_states"] is raw["last_hidden_states"]
    assert "last_hidden_states" not in adapted


def test_adapt_feature_rejects_misaligned_tokens_and_mask():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="lengths differ"):
        _adapt_feature(
            {
                "input_ids": torch.arange(3),
                "loss_mask": torch.ones(2),
                "hidden_states": torch.zeros(1, 3, 4),
                "last_hidden_states": torch.ones(1, 3, 2),
            }
        )


def test_finite_float_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        _finite_float(float("nan"), "metric")
