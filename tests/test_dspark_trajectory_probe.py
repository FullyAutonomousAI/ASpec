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

from tools.validation.dspark_confidence_probe import confidence_weights, projected_gradient
from tools.validation.dspark_nonpacking_operator_probe import interval_mask, norm_rope
from tools.validation.dspark_trajectory_probe import compare_maps, load_state, load_target_head


def test_confidence_weights_preserve_prefix_supervision_and_invalid_anchors():
    anchors = torch.zeros(1, 512, dtype=torch.long)
    anchors[0, :2] = torch.tensor([2, 10])
    keep = torch.zeros_like(anchors, dtype=torch.bool)
    keep[0, :2] = True
    mask = torch.ones(1, 12)
    mask[0, 5] = 0
    actual = confidence_weights(
        dict(
            input_ids=torch.zeros(1, 12, dtype=torch.long),
            loss_mask=mask,
            injected_anchors=anchors,
            injected_keep_mask=keep,
        )
    )
    expected = torch.zeros(1, 512, 7)
    expected[0, 0, :2] = torch.exp(-torch.arange(2) / 4)
    expected[0, 1, 0] = 1
    assert torch.equal(actual, expected)


def test_projected_gradient_is_pooled_not_a_mean_of_means():
    actual = projected_gradient(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.tensor([0.1, -0.2]), torch.tensor([1.0, 3.0])
    )
    torch.testing.assert_close(actual, torch.tensor([0.025, -0.3]))


def test_gradient_comparison_rejects_identity_and_presence_mismatch():
    with pytest.raises(ValueError, match="identities"):
        compare_maps({"a": torch.ones(1)}, {"b": torch.ones(1)})
    with pytest.raises(ValueError, match="presence"):
        compare_maps({"a": torch.ones(1)}, {"a": None})
    assert compare_maps({"a": torch.ones(1)}, {"a": torch.ones(1)})["global"]["relative_l2"] == 0


def test_hf_initialization_preserves_native_wrapper_keys(tmp_path):
    from safetensors.torch import save_file

    save_file({"layer.weight": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    assert list(load_state(tmp_path)) == ["draft_model.layer.weight"]


@pytest.mark.parametrize("engine,prenorm", [("sglang", False), ("hf", False), ("vllm", True)])
def test_probe_target_and_norm_come_from_fixture(tmp_path, monkeypatch, engine, prenorm):
    from types import SimpleNamespace

    from aspec.models.target.target_utils import TargetLMHead

    (tmp_path / "fast.yaml").write_text(
        "model:\n  target_model_path: /test/qwen-target\n"
        f"inference:\n  inference_engine_type: {engine}\n"
    )
    observed = []
    head = SimpleNamespace()

    def load(target, **options):
        observed.append((target, options))
        return head

    monkeypatch.setattr(TargetLMHead, "from_pretrained", load)
    assert load_target_head(tmp_path) is head
    assert observed == [("/test/qwen-target", {"load_norm": prenorm})]
    assert head._aspec_teacher_contract["last_hidden_states_prenorm"] is prenorm


def test_probe_rejects_unknown_teacher_stage_before_loading_weights(tmp_path, monkeypatch):
    from aspec.models.target.target_utils import TargetLMHead

    (tmp_path / "fast.yaml").write_text(
        "model:\n  target_model_path: /test/qwen-target\n"
        "inference:\n  inference_engine_type: offline\n"
    )

    def unexpected_load(*args, **kwargs):
        pytest.fail("Unknown normalization must fail before loading Target weights")

    monkeypatch.setattr(TargetLMHead, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="explicit"):
        load_target_head(tmp_path)


def test_math_oracle_has_exact_visibility_and_empty_query_rows():
    endpoints = torch.tensor([[2, 0], [3, 0], [5, 0]])
    assert torch.equal(
        interval_mask(endpoints, 7),
        torch.tensor([[True, True, False, True, True, False, False], [False] * 7]),
    )
    with pytest.raises(ValueError, match="three-endpoint"):
        interval_mask(torch.zeros(5, 2), 7)


def test_math_oracle_normalizes_and_rotates_indexed_positions():
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
    actual = norm_rope(
        value,
        torch.ones(4),
        torch.tensor([[0, 1]]),
        torch.tensor([[1.0] * 4, [0.0] * 4]).view(1, 1, 2, 4),
        torch.tensor([[0.0] * 4, [1.0] * 4]).view(1, 1, 2, 4),
        1,
        4,
        0,
    )
    expected = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[-3.0, -4.0, 1.0, 2.0]]]]) / (7.5**0.5)
    torch.testing.assert_close(actual, expected)
