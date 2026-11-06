# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Batched position metadata must satisfy the fused adapter's layout contract."""

import pytest
import torch

from aspec.models.dflash import DFlashModel


@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_batched_positions_are_compact_without_changing_values(batch, explicit_positions):
    model = DFlashModel.__new__(DFlashModel)
    model.block_size = 3
    anchors = torch.tensor([[1, 5]]).expand(batch, -1)
    base = torch.arange(12).unsqueeze(0).expand(batch, -1)
    context, query = model._create_position_ids(
        anchors, 12, base_position_ids=base if explicit_positions else None,
    )
    assert context.is_contiguous()
    assert query.is_contiguous()
    assert torch.equal(context, base)
    assert torch.equal(query, torch.tensor([[1, 2, 3, 5, 6, 7]]).expand(batch, -1))
