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

"""Repack a fixed optimizer document window with exact per-document anchors.

Quality-control instrumentation, not an online scheduler or a throughput tool.
The caller fixes document membership before changing physical execution. No
document is split, dropped, repeated or moved across an optimizer boundary.
"""


def pack_replay_documents(documents, capacity, *, num_anchors=512, block_size=7):
    """Use the production CP1 collator and preserve every kept anchor/label.

    Native BSHD reserves num_anchors slots even for short documents. THD's
    sample-scoped mode removes only structurally impossible, masked slots.
    Local anchors become packed-row offsets; document-local RoPE positions
    and attention boundaries still come from the production collator.
    """
    import torch

    from aspec.data.utils import DFlashPackingCollator
    from aspec.models.ops.anchors import document_anchor_slots

    if not documents or capacity < 1:
        raise ValueError("Expected a nonempty document window and positive capacity")
    groups, pending, used = [], [], 0
    for row in documents:
        ids = row["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or row.get("ctx_doc_ids") is not None:
            raise ValueError("Replay input must contain individual, unpacked documents")
        length = ids.shape[1]
        if not 0 < length <= capacity:
            raise ValueError("A whole document does not fit the packed capacity")
        if row["injected_anchors"].shape != (1, num_anchors):
            raise ValueError("Source anchor budget differs from the agreed document budget")
        if row["injected_keep_mask"].shape != (1, num_anchors):
            raise ValueError("Source anchor keep-mask shape differs")
        if pending and used + length > capacity:
            groups.append(pending)
            pending, used = [], 0
        pending.append(row)
        used += length
    groups.append(pending)

    result = []
    collator = DFlashPackingCollator(capacity, pad_to_max_length=False)
    occurrence = 0
    for group in groups:
        features = []
        for row in group:
            features.append({
                "input_ids": row["input_ids"],
                "loss_mask": row["loss_mask"],
                "hidden_states": torch.cat(row["hidden_states_list"], dim=-1),
                "last_hidden_states": row["last_hidden_states"],
                # Collator-only replay identity. These owned CPU copies never
                # enter the live controller's storage or committed-work ledger.
                "_aspec_work": {"work_id": f"replay-{occurrence}",
                                "execution_id": f"replay-{occurrence}"},
            })
            occurrence += 1
        collated = collator(features)
        lengths = collated["packed_document_lengths"]
        slots, _, padded = document_anchor_slots(lengths, num_anchors, block_size)
        anchors, keeps, offset = [], [], 0
        for row, length, count in zip(group, lengths, slots, strict=True):
            keep = row["injected_keep_mask"]
            if keep.dtype != torch.bool or keep[:, count:].any():
                raise ValueError("Compaction would remove a kept source anchor")
            kept = keep[:, :count]
            anchors.append(torch.where(kept, row["injected_anchors"][:, :count] + offset, 0))
            keeps.append(kept)
            offset += length
        values = {
            key: collated[key] for key in (
                "input_ids", "loss_mask", "ctx_doc_ids", "base_position_ids",
                "last_hidden_states", "packed_document_lengths", "attention_mask",
            )
        }
        width = group[0]["hidden_states_list"][0].shape[-1]
        values["hidden_states_list"] = list(collated["hidden_states"].split(width, dim=-1))
        anchor = torch.cat(anchors, dim=1)
        keep = torch.cat(keeps, dim=1)
        values["injected_anchors"] = torch.nn.functional.pad(anchor, (0, padded-anchor.shape[1]))
        values["injected_keep_mask"] = torch.nn.functional.pad(keep, (0, padded-keep.shape[1]))
        result.append(values)
    return result
