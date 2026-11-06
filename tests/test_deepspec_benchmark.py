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

from tools.performance.deepspec_benchmark import (
    adapt_feature,
    adapt_state,
    resident_batch,
)


def test_initial_adapter_only_renames_three_tensors():
    state = {name: object() for name in (
        "context_proj.weight", "context_norm.weight", "final_norm.weight", "embed_tokens.weight",
    )}
    converted = adapt_state(state)
    assert converted == {
        "fc.weight": state["context_proj.weight"],
        "hidden_norm.weight": state["context_norm.weight"],
        "norm.weight": state["final_norm.weight"],
        "embed_tokens.weight": state["embed_tokens.weight"],
    }


def test_initial_adapter_rejects_missing_or_ambiguous_keys():
    with pytest.raises(ValueError, match="missing"):
        adapt_state({})
    with pytest.raises(ValueError, match="collisions"):
        adapt_state({key: object() for key in (
            "context_proj.weight", "context_norm.weight", "final_norm.weight", "fc.weight",
        )})


def feature():
    import torch

    return {
        "input_ids": torch.arange(3)[None], "loss_mask": torch.ones(1, 3),
        "hidden_states": torch.zeros(1, 3, 10), "last_hidden_states": torch.zeros(1, 3, 2),
    }


def test_feature_adapter_keeps_values_and_normalized_teacher():
    raw = feature()
    converted = adapt_feature(raw)
    assert converted["input_ids"].shape == (3,)
    assert converted["target_hidden_states"].shape == (3, 10)
    assert converted["target_last_hidden_states"].shape == (3, 2)
    assert converted["target_hidden_states"].data_ptr() == raw["hidden_states"].data_ptr()
    assert converted["target_last_hidden_states"].data_ptr() == raw["last_hidden_states"].data_ptr()


def test_feature_adapter_rejects_batch_or_length_mismatch():
    import torch

    raw = feature()
    raw["input_ids"] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="shape"):
        adapt_feature(raw)
    raw = feature()
    raw["hidden_states"] = torch.zeros(1, 4, 10)
    with pytest.raises(ValueError, match="lengths differ"):
        adapt_feature(raw)


def test_feature_adapter_requires_complete_teacher_features():
    with pytest.raises(ValueError, match="last_hidden_states"):
        adapt_feature({})


def test_resident_batch_uses_views_without_copying_features():
    raw = adapt_feature(feature())
    batch = resident_batch(raw)
    for key, value in raw.items():
        assert batch[key].shape == (1, *value.shape)
        assert batch[key].data_ptr() == value.data_ptr()
        assert batch[key].device == value.device
