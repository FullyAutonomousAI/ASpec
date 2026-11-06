# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Physical BSHD padding must not change document-local anchor candidates."""

import pytest
import torch
import torch.nn.functional as F

from aspec.data.utils import DataCollatorWithPadding
from aspec.models.dflash import DFlashModel
from aspec.models.ops.anchors import sample_anchor_positions


def _items(lengths):
    return [{"input_ids": torch.arange(n).view(1, n),
             "loss_mask": torch.ones(1, n),
             "hidden_states": torch.randn(1, n, 8)} for n in lengths]


@pytest.mark.parametrize("lengths", [(133,), (19, 133), (8, 19, 133, 255)])
def test_collator_uses_exact_longest_document(lengths):
    batch = DataCollatorWithPadding()(_items(lengths))
    assert batch["input_ids"].shape == (len(lengths), max(lengths))
    assert batch["hidden_states"].shape == (len(lengths), max(lengths), 8)
    assert batch["attention_mask"].sum(1).tolist() == list(lengths)
    assert batch["loss_mask"].sum(1).tolist() == list(lengths)


@pytest.mark.parametrize("extra_padding", [0, 1, 123, 257])
@pytest.mark.parametrize("scope", ["row", "sample"])
def test_anchor_domains_match_unpadded_documents_regardless_of_batch_padding(extra_padding, scope):
    lengths = (7, 19, 133, 255)
    items = _items(lengths)
    items[2]["loss_mask"][:, 20:34] = 0
    batch = DataCollatorWithPadding()(items)
    loss_mask = F.pad(batch["loss_mask"], (0, extra_padding))
    mask = F.pad(batch["attention_mask"], (0, extra_padding))
    wrapper = DFlashModel(torch.nn.Module(), block_size=7, num_anchors=512,
                          anchor_sampling_scope=scope)
    torch.manual_seed(19)
    anchors, keep = wrapper._sample_anchor_positions(
        loss_mask.shape[1], loss_mask, loss_mask.device, attention_mask=mask)
    torch.manual_seed(19)
    expected = []
    for n, item in zip(lengths, items, strict=True):
        # The vectorized batch reserves one phase for every row, including an
        # empty short row. The native singleton returns before drawing that phase.
        if n <= 7:
            torch.rand(1)
        expected.append(sample_anchor_positions(n, item["loss_mask"], 512, 7, loss_mask.device))
    for i, (ref_anchors, ref_keep) in enumerate(expected):
        torch.testing.assert_close(anchors[i], ref_anchors[0])
        torch.testing.assert_close(keep[i], ref_keep[0])


def test_cached_document_padding_retains_its_validity_mask():
    item = _items((133,))[0]
    item["input_ids"] = F.pad(item["input_ids"], (0, 123))
    item["loss_mask"] = F.pad(item["loss_mask"], (0, 123))
    item["attention_mask"] = F.pad(torch.ones(1, 133), (0, 123))
    item["hidden_states"] = F.pad(item["hidden_states"], (0, 0, 0, 123))
    batch = DataCollatorWithPadding()([item])
    assert batch["attention_mask"].sum() == 133
    assert batch["loss_mask"].sum() == 133
