# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from aspec.models.ops.anchors import validate_injected_anchors
from tools.validation.dspark_layout_alignment import pack_replay_documents


def document(length, identity):
    ids = torch.full((1, length), identity, dtype=torch.long)
    anchors = torch.zeros(1, 8, dtype=torch.long)
    keep = torch.zeros(1, 8, dtype=torch.bool)
    count = min(8, max(length-6, 0)) if length > 7 else 0
    anchors[0, :count] = torch.arange(count)
    keep[0, :count] = True
    return dict(input_ids=ids, loss_mask=torch.ones_like(ids).float(),
                hidden_states_list=[ids.unsqueeze(-1).expand(-1, -1, 2).float()],
                last_hidden_states=ids.unsqueeze(-1).expand(-1, -1, 2).float(),
                injected_anchors=anchors, injected_keep_mask=keep)


def test_repacking_preserves_document_content_local_positions_and_anchor_targets():
    docs = [document(n, i) for i, n in enumerate((9, 13, 20))]
    rows = pack_replay_documents(docs, 25, num_anchors=8)
    assert [row["packed_document_lengths"] for row in rows] == [(9, 13), (20,)]
    assert torch.equal(torch.cat([row["input_ids"] for row in rows], 1),
                       torch.cat([row["input_ids"] for row in docs], 1))
    row = rows[0]
    assert row["base_position_ids"].tolist() == [list(range(9)) + list(range(13))]
    assert row["injected_anchors"].tolist() == [[0, 1, 2, 9, 10, 11, 12, 13, 14, 15]]
    assert all(row["injected_keep_mask"].flatten())
    validate_injected_anchors(row["injected_anchors"], row["injected_keep_mask"],
                             loss_mask=row["loss_mask"], num_anchors=10, block_size=7,
                             ctx_doc_ids=row["ctx_doc_ids"], allow_partial_last_target=True)
    # The final prediction of the first document's last anchor must stay masked,
    # although the next document physically follows it in the packed row.
    last_label = row["injected_anchors"][0, 2] + 7
    assert row["ctx_doc_ids"][0, last_label] != row["ctx_doc_ids"][0, 2]


def test_short_document_dummy_slots_cannot_consume_another_document_budget():
    rows = pack_replay_documents([document(7, 1), document(9, 2)], 16, num_anchors=8)
    assert rows[0]["injected_anchors"].tolist() == [[7, 8, 9]]
    assert rows[0]["injected_keep_mask"].sum() == 3
    empty = pack_replay_documents([document(7, 1)], 16, num_anchors=8)[0]
    assert empty["injected_anchors"].shape == (1, 1)
    assert not empty["injected_keep_mask"].any()


def test_repacking_rejects_overflow_and_lossy_compaction():
    with pytest.raises(ValueError, match="does not fit"):
        pack_replay_documents([document(20, 1)], 19, num_anchors=8)
    row = document(9, 1)
    row["injected_keep_mask"][0, -1] = True
    with pytest.raises(ValueError, match="remove a kept"):
        pack_replay_documents([row], 20, num_anchors=8)
