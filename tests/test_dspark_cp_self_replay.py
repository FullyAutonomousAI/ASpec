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

from tools.validation.dspark_cp_self_replay import owner_inputs


def test_replay_uses_real_per_document_cp_ownership_without_losing_any_token():
    values = torch.arange(12).view(1, 12, 1).bfloat16()
    row = {
        "input_ids": torch.arange(12).view(1, -1),
        "loss_mask": torch.ones(1, 12),
        "ctx_doc_ids": torch.tensor([[0] * 5 + [1] * 7]),
        "base_position_ids": torch.tensor([list(range(5)) + list(range(7))]),
        "hidden_states_list": [values, values + 20],
        "last_hidden_states": values + 40,
        "injected_anchors": torch.tensor([[0, 5]]),
        "injected_keep_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    for size in (2, 3, 4):
        owners = [owner_inputs(row, rank, size) for rank in range(size)]
        positions = torch.cat([item["packed_context_positions"][0] for item in owners])
        assert positions.unique().numel() == 12
        assert torch.equal(positions.sort().values, torch.arange(12))
        for item in owners:
            indices = item["packed_context_positions"][0]
            for source, actual in zip(
                row["hidden_states_list"], item["hidden_states_list"], strict=True
            ):
                assert torch.equal(actual, source.index_select(1, indices))
            assert torch.equal(
                item["last_hidden_states"], row["last_hidden_states"].index_select(1, indices)
            )
            assert torch.equal(item["injected_anchors"], row["injected_anchors"])
