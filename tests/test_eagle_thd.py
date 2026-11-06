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

import torch

from aspec.models.ops.eagle_thd import EagleTHDLayout


def batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "loss_mask": torch.ones(2, 4),
        "hidden_states": torch.arange(16).reshape(2, 4, 2).float(),
        "last_hidden_states": torch.arange(16).reshape(2, 4, 2).float() + 100,
    }


def test_depth_shifts_never_cross_document_or_supervise_fake_teacher():
    layout, hidden, teacher = EagleTHDLayout.from_batch(batch())
    assert hidden.shape == (1, 6, 2)
    ids, mask = layout.depth_inputs(0)
    assert ids.tolist() == [[2, 3, 4, 0, 6, 0]]
    assert mask.tolist() == [[1, 1, 1, 0, 1, 0]]
    ids, mask = layout.depth_inputs(1)
    assert ids.tolist() == [[3, 4, 0, 0, 0, 0]]
    assert mask.tolist() == [[1, 1, 0, 0, 0, 0]]
    actual = layout.teacher_at(teacher, 0)
    torch.testing.assert_close(actual[0, :3], teacher[0, 1:4])
    assert actual[0, 3].count_nonzero() == 0
    torch.testing.assert_close(actual[0, 4], teacher[0, 5])
    assert actual[0, 5].count_nonzero() == 0


def test_cp_geometry_is_not_a_degree_whitelist_and_can_have_empty_owners():
    value = batch()
    for rank in range(7):
        value["hidden_states"] = torch.zeros(2, 1, 2)
        value["last_hidden_states"] = torch.zeros(2, 1, 2)
        layout, hidden, _ = EagleTHDLayout.from_batch(value, cp_rank=rank, cp_size=7)
        assert hidden.shape[1] == layout.local_length
        assert layout.owned in (0, 1, 2)
        assert layout.global_positions[layout.global_positions >= 0].sort().values.tolist() == list(range(6))


def test_runtime_attention_metadata_is_reused_only_within_one_batch():
    layout, _, _ = EagleTHDLayout.from_batch(batch())
    first = layout.attention_aux(2, False)
    assert layout.attention_aux(2, False) is first
    assert first[2].shape[-1] == 18
    other, _, _ = EagleTHDLayout.from_batch(batch())
    assert other.attention_aux(2, False) is not first

def test_packing_work_value_is_ttt_supervision_not_unused_block_anchors():
    from aspec.controller.packing import count_eagle_supervised_positions

    assert count_eagle_supervised_positions([0, 5], 5, 3) == 4 + 3 + 2
    assert count_eagle_supervised_positions("3,2", 5, 3) == 3
    assert count_eagle_supervised_positions([5], 5, 3) == 0
