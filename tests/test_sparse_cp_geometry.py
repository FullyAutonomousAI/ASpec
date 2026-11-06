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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from aspec.models.draft.csa import (
    CSAForwardMetadata,
    CSALightningIndexer,
    _csa_remote_predecessor_halo,
    build_csa_compression_layout,
)


@pytest.mark.parametrize("stage", ["warmup", "sparse"])
def test_csa_uneven_owners_agree_on_collective_chunks(stage):
    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        compress_ratios=[4],
        compress_rope_theta=40000.0,
        rms_norm_eps=1e-6,
    )
    schedules = []
    for length in [12, 16]:
        positions = torch.arange(length)[None]
        metadata = CSAForwardMetadata(
            anchor_positions=torch.tensor([[1, 9, 17, 25]]),
            block_keep_mask=torch.ones(1, 4, dtype=torch.bool),
            block_size=2,
            context_window=4,
            context_physical_position_ids=positions,
        )
        layout = build_csa_compression_layout(metadata, positions, ratio=4)
        direct = SimpleNamespace(
            cp_size=2,
            group=object(),
            sequence_length=28,
            collective_block_ids=torch.arange(4),
            layout="contiguous",
            _fa4_cache={},
        )
        chunks = []

        def teacher(**kw):
            count = kw["query"].shape[1]
            chunks.append(count)
            width = layout.num_groups if stage == "warmup" else 2
            return torch.zeros(1, count, width), torch.ones(1, count, dtype=torch.bool)

        def winners(scores, eligible, **kw):
            shape = (1, scores.shape[1], 2)
            return SimpleNamespace(
                local_winner_indices=torch.full(shape, -1, dtype=torch.long),
                scores=torch.zeros(shape),
                valid=torch.ones(shape, dtype=torch.bool),
            )

        with (
            patch(
                "aspec.models.ops.direct_cp.all_gather_query_rows",
                side_effect=lambda x, **kw: torch.cat((x, x), dim=1),
            ),
            patch(
                "aspec.models.ops.distributed_sparse.distributed_log_softmax",
                side_effect=lambda s, e, **kw: (s * 0, e.any(-1)),
            ),
            patch("aspec.models.ops.distributed_sparse.distributed_topk", side_effect=winners),
            patch("aspec.models.draft.csa._distributed_csa_full_target", side_effect=teacher),
            patch("aspec.models.draft.csa._distributed_csa_selected_target", side_effect=teacher),
            patch("torch.distributed.nn.functional.all_reduce", side_effect=lambda x, **kw: x),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            indexer = CSALightningIndexer(config, 0)
            call = (
                indexer.forward_distributed_warmup
                if stage == "warmup"
                else indexer.forward_distributed_sparse
            )
            call(
                draft_hidden=torch.randn(1, 4, 16),
                context_hidden=torch.randn(1, length, 16),
                draft_position_ids=torch.tensor([[1, 2, 9, 10]]),
                layout=layout,
                metadata=metadata,
                main_query=torch.randn(1, 4, 4, 4),
                main_compressed_key=torch.randn(1, layout.num_groups, 2, 4),
                scale=0.5,
                workspace_bytes=150,
                direct_cp_metadata=direct,
            )
        schedules.append(chunks)
    assert schedules[0] == schedules[1], schedules


def test_csa_empty_owner_still_participates_in_halo_request():
    anchors = torch.tensor([[1, 3]])
    positions = torch.empty(1, 0, dtype=torch.long)
    metadata = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=1,
        context_window=4,
        context_physical_position_ids=positions,
    )
    layout = build_csa_compression_layout(metadata, positions, ratio=4)
    direct = SimpleNamespace(
        cp_size=2, group=object(), sequence_length=4, context_position_ids=positions, _fa4_cache={}
    )

    def gather(hidden, owned, requested, **kw):
        assert requested.shape == (1, 4)
        assert requested.eq(-1).all()
        return hidden.new_zeros(1, 4, 8)

    with patch(
        "aspec.models.ops.direct_cp.gather_owner_sharded_rows", side_effect=gather
    ) as exchange:
        _csa_remote_predecessor_halo(
            context_hidden=torch.empty(1, 0, 8),
            metadata=metadata,
            layout=layout,
            ratio=4,
            direct_cp_metadata=direct,
        )
    assert exchange.call_count == 1


def test_packed_workspace_uses_actual_owner_capacity_once():
    from aspec.models.ops.distributed_sparse import max_context_shard_length

    # Three four-token documents at CP2/alignment4: rank0 owns zero rows,
    # rank1 owns all 12. Sharding the packed total would predict only eight.
    meta = SimpleNamespace(
        cp_size=2,
        group=object(),
        sequence_length=12,
        layout="contiguous",
        context_doc_ids=torch.empty(1, 0, dtype=torch.long),
        context_position_ids=torch.empty(1, 0, dtype=torch.long),
        _fa4_cache={},
    )
    with patch(
        "torch.distributed.all_reduce", side_effect=lambda capacity, **kw: capacity.fill_(12)
    ) as reduce:
        assert max_context_shard_length(meta, 0, alignment=4) == 12
        assert max_context_shard_length(meta, 0, alignment=4) == 12
    assert reduce.call_count == 1


@pytest.mark.parametrize("kind", ["qsa", "csa"])
def test_sparse_rows_allow_empty_packed_owner(kind):
    from aspec.models.draft.csa import (
        CSAForwardMetadata,
        _distributed_csa_token_rows,
        build_csa_compression_layout,
    )
    from aspec.models.draft.qsa import (
        QSAForwardMetadata,
        _distributed_qsa_token_rows,
        build_qsa_compression_layout,
    )
    from aspec.models.ops.direct_cp import DirectCPMetadata

    positions = torch.empty(1, 0, dtype=torch.long)
    anchors = torch.tensor([[1, 3]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    docs = torch.zeros_like(anchors)
    common = dict(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=1,
        context_doc_ids=positions,
        anchor_doc_ids=docs,
        context_physical_position_ids=positions,
    )
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=1,
        context_position_ids=positions,
        context_doc_ids=positions,
        anchor_doc_ids=docs,
        cp_size=2,
        cp_rank=0,
        group=object(),
        sequence_length=4,
    )
    selected = torch.full((1, 2, 1), -1, dtype=torch.long)
    if kind == "qsa":
        meta = QSAForwardMetadata(**common)
        layout = build_qsa_compression_layout(meta, positions, 4)
        _, valid, _ = _distributed_qsa_token_rows(
            metadata=meta, layout=layout, local_selected=selected, direct_cp_metadata=direct
        )
    else:
        meta = CSAForwardMetadata(**common, context_window=4)
        layout = build_csa_compression_layout(meta, positions, 4)
        _, valid, _ = _distributed_csa_token_rows(
            metadata=meta,
            layout=layout,
            local_selected=selected,
            direct_cp_metadata=direct,
            query_indices=torch.arange(2),
            raw_key_length=1,
        )
    assert valid.sum().item() == 1
