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

"""CPU-only checks for FA4 fixed sparse tile dispatch; no kernel compilation."""

import sys
from types import SimpleNamespace

import pytest
import torch

from aspec.models.ops import fa4_sparse


def test_sm100_keeps_b27_backward_layout(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    assert fa4_sparse._fa4_sparse_block_sizes(torch.device("cuda"), 128, 128) == (
        (256, 128),
        (256, 128),
    )


@pytest.mark.parametrize("builder", ["fixed", "direct_cp"])
@pytest.mark.parametrize("head_dim,fwd_kv,bwd_kv", [(64, 128, 128), (128, 128, 128), (256, 80, 64)])
def test_sm90_pair_uses_independent_fa4_dispatch(monkeypatch, builder, head_dim, fwd_kv, bwd_kv):
    calls = []

    def tile(direction, kv_tile):
        def select(query_dim, value_dim, causal, local, *, sparse_block_size_q):
            calls.append((direction, query_dim, value_dim, causal, local, sparse_block_size_q))
            return SimpleNamespace(n_block_size=kv_tile)

        return select

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute.interface",
        SimpleNamespace(
            _tile_size_fwd_sm90=tile("forward", fwd_kv),
            _tile_size_bwd_sm90=tile("backward", bwd_kv),
        ),
    )
    monkeypatch.setattr(fa4_sparse, "_build_fixed_sparse", lambda **kwargs: kwargs["block_size"])
    monkeypatch.setattr(fa4_sparse, "_transpose_fixed_sparse", lambda source, size: source)
    monkeypatch.setattr(fa4_sparse, "_direct_cp_sparse", lambda **kwargs: kwargs["block_size"])
    common = dict(query_head_dim=head_dim, value_head_dim=head_dim)
    if builder == "fixed":
        pair = fa4_sparse.build_fixed_sparse_pair(
            **common,
            mask_mod=lambda: None,
            aux_tensors=[],
            batch_size=1,
            query_length=256,
            key_length=512,
            device=torch.device("cuda"),
        )
    else:
        tensor = torch.zeros(1, 1)
        pair = fa4_sparse.build_direct_cp_sparse_pair(
            **common,
            anchor_positions=tensor,
            query_blocks=tensor,
            context_positions=tensor,
            draft_blocks=tensor,
            query_token_keep=tensor,
            key_token_keep=tensor,
        )
    assert pair.forward == (256, fwd_kv)
    assert pair.backward == (256, bwd_kv)
    assert calls == [
        (direction, head_dim, head_dim, False, False, 256) for direction in ("forward", "backward")
    ]
