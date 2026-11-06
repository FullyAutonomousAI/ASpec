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

"""Cheap dispatch regressions and opt-in exact CUDA metadata comparisons."""

import os
from types import SimpleNamespace

import pytest
import torch

from aspec.models.ops import fa4_sparse, thd_attention, thd_sparse_metadata
from aspec.models.ops.direct_cp import _visibility


def test_actual_lengths_do_not_specialize_a_kernel_per_packed_row():
    kernel = thd_sparse_metadata._classify_tiles
    constants = {kernel.arg_names[index] for index in kernel.constexprs}
    assert not {"Q", "K", "NM", "NN"} & constants
    assert {"BL", "TM", "TN", "QM", "KN"} <= constants


@pytest.mark.parametrize("forward,backward", [((256, 128), (256, 128)), ((256, 80), (256, 64))])
def test_pair_respects_independent_tiles_and_reuses_equal_geometry(monkeypatch, forward, backward):
    calls = []
    monkeypatch.setattr(fa4_sparse, "_fa4_sparse_block_sizes", lambda *args: (forward, backward))
    monkeypatch.setattr(fa4_sparse, "_transpose_fixed_sparse", lambda source, geometry: source)

    def build(**kwargs):
        calls.append((kwargs["tile_m"], kwargs["tile_n"]))
        return calls[-1]

    monkeypatch.setattr(thd_sparse_metadata, "_build_sparse_source", build)
    pair = thd_sparse_metadata.build_thd_sparse_pair(
        aux_tensors=[SimpleNamespace(device="cuda")],
        block_size=7,
        query_length=560,
        key_length=1099,
        query_head_dim=128,
        value_head_dim=128,
    )
    assert pair.forward == forward and pair.backward == backward
    assert calls == ([forward] if forward == backward else [forward, backward])


def test_invalid_mask_layout_is_rejected():
    with pytest.raises(ValueError, match="eight THD"):
        thd_sparse_metadata._build_sparse_source(
            aux_tensors=[],
            block_size=7,
            query_length=7,
            key_length=10,
            tile_m=256,
            tile_n=128,
        )


def runtime_length_gpu_check():
    """Exact independent visibility, without compiling the generic FA4 oracle."""
    from aspec.models.ops.direct_cp import DirectCPMetadata

    device = torch.device("cuda")
    for length in (539, 575):
        docs = torch.cat([torch.full((n,), i, device=device)
                          for i, n in enumerate((257, 263, length - 520))])
        anchors = torch.tensor([[0, 19, 256, 257, 281, 519, 520, 537]], device=device)
        for block_size in (7, 257):
            for query_window, causal in ((None, False), (5, True)):
                meta = DirectCPMetadata(
                    anchors, torch.tensor([[True, True, False, True, True, True, True, True]], device=device),
                    torch.arange(length, device=device)[None], block_size, 0, 1, None,
                    context_doc_ids=docs[None], anchor_doc_ids=docs[anchors], sequence_length=length,
                )
                meta.query_window, meta.draft_is_causal = query_window, causal
                qdocs, qblocks, qanchors, qkeep = thd_attention._query_metadata(meta, device)
                kdocs, kp, kb, kk = thd_attention._key_metadata(meta, device)
                aux = [t.int()[None].contiguous() for t in
                       (qanchors, qkeep, qblocks, qdocs, kdocs, kp, kb, kk)]
                expected = _visibility(meta, device)[0]
                expected &= (qdocs[:, None] == kdocs[None, :]) & qkeep[:, None] & kk[None, :]
                for tm, tn in ((256, 64), (256, 80), (256, 128)):
                    actual = thd_sparse_metadata._build_sparse_source(
                        aux_tensors=aux, block_size=block_size, query_length=qanchors.numel(),
                        key_length=kp.numel(), tile_m=tm, tile_n=tn,
                        query_window=query_window, draft_is_causal=causal,
                    )
                    def tiles(fill):
                        matrix = torch.nn.functional.pad(
                            expected, (0, (-expected.shape[1]) % tn, 0, (-expected.shape[0]) % tm),
                            value=fill,
                        )
                        return matrix.reshape(-1, tm, matrix.shape[1] // tn, tn).permute(0, 2, 1, 3)
                    columns = actual.mask_block_idx.shape[-1]
                    partial = fa4_sparse._ordered_to_dense(
                        actual.mask_block_cnt, actual.mask_block_idx, columns)[0, 0]
                    full = fa4_sparse._ordered_to_dense(
                        actual.full_block_cnt, actual.full_block_idx, columns)[0, 0]
                    torch.testing.assert_close(partial | full, tiles(False).any(dim=(-1, -2)), atol=0, rtol=0)
                    torch.testing.assert_close(full, tiles(True).all(dim=(-1, -2)), atol=0, rtol=0)
    print("PASS runtime Q/K bounds: independent exact visibility, document/tail, SWA/causal, 64/80/128 tiles")


@pytest.mark.skipif(os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1", reason="opt-in runtime metadata")
def test_runtime_sequence_bounds_match_independent_visibility():
    runtime_length_gpu_check()


def classification_gpu_check():
    # Unequal packed documents; a disabled block, a missing doc, and a partial
    # last block. BL=257 also intersects a tile without a block boundary.
    from aspec.models.ops.direct_cp import DirectCPMetadata

    device = torch.device("cuda")
    docs = torch.cat(
        [torch.full((length,), index, device=device) for index, length in enumerate([257, 263, 19])]
    )
    for block_size in (1, 7, 31, 257):
        anchors = torch.tensor([[0, 19, 256, 257, 281, 519, 520, 537]], device=device)
        anchor_docs = docs[anchors].clone()
        anchor_docs[0, 6] = -1
        meta = DirectCPMetadata(
            anchors,
            torch.tensor([[True, True, False, True, True, True, True, True]], device=device),
            torch.arange(docs.numel(), device=device)[None],
            block_size,
            0,
            1,
            None,
            context_doc_ids=docs[None],
            anchor_doc_ids=anchor_docs,
            sequence_length=docs.numel(),
        )
        for query_window, causal in ((None, False), (5, False), (5, True), (None, True)):
            meta.query_window, meta.draft_is_causal = query_window, causal
            qdocs, qblocks, qanchors, qkeep = thd_attention._query_metadata(meta, device)
            kdocs, kpositions, kblocks, kkeep = thd_attention._key_metadata(meta, device)
            aux = [
                tensor.int()[None].contiguous()
                for tensor in (qanchors, qkeep, qblocks, qdocs, kdocs, kpositions, kblocks, kkeep)
            ]
            for tile in ((256, 128), (256, 80), (256, 64)):
                actual = thd_sparse_metadata._build_sparse_source(
                    aux_tensors=aux,
                    block_size=block_size,
                    query_length=qanchors.numel(),
                    key_length=kpositions.numel(),
                    tile_m=tile[0],
                    tile_n=tile[1], query_window=query_window, draft_is_causal=causal,
                )
                generic = fa4_sparse._build_fixed_sparse(
                    mask_mod=thd_attention._build_fixed_mask(query_window=query_window,
                        draft_is_causal=causal, block_size=block_size),
                    aux_tensors=aux,
                    batch_size=1,
                    query_length=qanchors.numel(),
                    key_length=kpositions.numel(),
                    device=device,
                    block_size=tile,
                )
                columns = actual.mask_block_idx.shape[-1]
                for name in ("mask", "full"):
                    # Unused slots in FA4's packed indices are unspecified.
                    ac, ai = getattr(actual, name + "_block_cnt"), getattr(actual, name + "_block_idx")
                    gc, gi = (
                        getattr(generic, name + "_block_cnt"),
                        getattr(generic, name + "_block_idx"),
                    )
                    torch.testing.assert_close(ac, gc, rtol=0, atol=0)
                    torch.testing.assert_close(
                        fa4_sparse._ordered_to_dense(ac, ai, columns),
                        fa4_sparse._ordered_to_dense(gc, gi, columns),
                        rtol=0,
                        atol=0,
                    )

                independent = _visibility(meta, device)[0]
                independent &= (qdocs[:, None] == kdocs[None, :]) & qkeep[:, None] & kkeep[None, :]
                qpad = (-independent.shape[0]) % tile[0]
                kpad = (-independent.shape[1]) % tile[1]
                def tile_reduce(padding, reduce):
                    matrix = torch.nn.functional.pad(independent, (0, kpad, 0, qpad), value=padding)
                    tiles = matrix.reshape(-1, tile[0], matrix.shape[1] // tile[1], tile[1]).permute(0, 2, 1, 3)
                    return reduce(tiles, dim=(-1, -2))
                partial = fa4_sparse._ordered_to_dense(
                    actual.mask_block_cnt, actual.mask_block_idx, columns)[0, 0]
                full = fa4_sparse._ordered_to_dense(
                    actual.full_block_cnt, actual.full_block_idx, columns)[0, 0]
                torch.testing.assert_close(partial | full, tile_reduce(False, torch.any), atol=0, rtol=0)
                torch.testing.assert_close(full, tile_reduce(True, torch.all), atol=0, rtol=0)
    print(
        "PASS exact structured/generic THD sparse classes: BL1/7/31/257, FW/BWD tiles, causal/SWA, doc/tail/empty boundaries, independent visibility",
        flush=True,
    )


@pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1", reason="opt-in FA4 metadata check"
)
def test_structured_metadata_matches_generic_fa4():
    classification_gpu_check()


if __name__ == "__main__":
    classification_gpu_check()
