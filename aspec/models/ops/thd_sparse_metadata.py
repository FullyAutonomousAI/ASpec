# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from the NVIDIA Megatron-LM DSpark sparse-metadata implementation.
# Redistribution is governed by licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.

"""Exact fixed sparse traversal for THD's eight-tensor mask contract.

The producer repeats anchor, document and block identity for block_size query
tokens. Query keep is constant or a kept prefix followed by a masked tail.
Classify one representative per block intersection, checking its final keep
for full tiles. Keys remain token-exact, including CP ownership and doc edges.
The attention mask and FA4 forward/backward kernels are unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _classify_tiles(
    anchors,
    query_keep,
    query_blocks,
    query_docs,
    key_docs,
    key_positions,
    key_blocks,
    key_keep,
    classes,
    # Actual lengths and output strides are runtime bounds. Specializing them
    # compiles a new classifier for every packed document length, despite an
    # unchanged tile algorithm. Tile/visibility policy remains compile-time.
    Q,
    K,
    BL: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    QM: tl.constexpr,
    KN: tl.constexpr,
    NM,
    NN,
    WINDOW: tl.constexpr,
    QUERY_WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    batch = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)
    # Leading partial block and every subsequent block boundary in this tile.
    r = tl.arange(0, QM)
    begin = m * TM
    first_boundary = tl.cdiv(begin, BL) * BL
    qi = tl.where(r == 0, begin, first_boundary + (r - 1) * BL)
    valid_q = (qi < Q) & (qi < begin + TM)
    qend = tl.minimum((qi // BL + 1) * BL - 1, tl.minimum(begin + TM - 1, Q - 1))
    ki = n * TN + tl.arange(0, KN)
    valid_k = (ki < K) & (ki < (n + 1) * TN)

    qa = tl.load(anchors + batch * Q + qi, valid_q, other=0)
    qb = tl.load(query_blocks + batch * Q + qi, valid_q, other=0)
    qdoc = tl.load(query_docs + batch * Q + qi, valid_q, other=-1)
    qany = tl.load(query_keep + batch * Q + qi, valid_q, other=0) > 0
    qall = tl.load(query_keep + batch * Q + qend, valid_q, other=0) > 0
    kdoc = tl.load(key_docs + batch * K + ki, valid_k, other=-1)
    kp = tl.load(key_positions + batch * K + ki, valid_k, other=-1)
    kb = tl.load(key_blocks + batch * K + ki, valid_k, other=-1)
    kkeep = tl.load(key_keep + batch * K + ki, valid_k, other=0) > 0
    context_visible = (kp[None, :] >= 0) & (kp[None, :] < qa[:, None])
    if WINDOW > 0:
        context_visible &= kp[None, :] >= qa[:, None] - WINDOW
    draft_visible = (kb[None, :] >= 0) & (kb[None, :] == qb[:, None])
    context_full = context_visible
    draft_full = draft_visible
    qlo, qhi = qi % BL, qend % BL
    koff = -kp - 1
    if QUERY_WINDOW > 0:
        context_visible &= kp[None, :] > qa[:, None] + qlo[:, None] - QUERY_WINDOW
        context_full &= kp[None, :] > qa[:, None] + qhi[:, None] - QUERY_WINDOW
        draft_visible &= (koff[None, :] > qlo[:, None] - QUERY_WINDOW) & (
            koff[None, :] < qhi[:, None] + QUERY_WINDOW)
        draft_full &= (koff[None, :] > qhi[:, None] - QUERY_WINDOW) & (
            koff[None, :] < qlo[:, None] + QUERY_WINDOW)
    if CAUSAL:
        draft_visible &= koff[None, :] <= qhi[:, None]
        draft_full &= koff[None, :] <= qlo[:, None]
    base = (qdoc[:, None] == kdoc[None, :]) & kkeep[None, :]
    visible = (context_visible | draft_visible) & base & qany[:, None]
    fully_visible = (context_full | draft_full) & base & qall[:, None]
    valid = valid_q[:, None] & valid_k[None, :]
    any_visible = tl.max(tl.max((visible & valid).to(tl.int32), 1), 0) != 0
    any_masked = tl.max(tl.max((~fully_visible & valid).to(tl.int32), 1), 0) != 0
    result = tl.where(any_visible, tl.where(any_masked, 1, 2), 0)
    tl.store(classes + (batch * NM + m) * NN + n, result)


def _build_sparse_source(*, aux_tensors, block_size, query_length, key_length, tile_m, tile_n,
                         context_window=None, query_window=None, draft_is_causal=False):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    from aspec.models.ops.fa4_sparse import _dense_to_ordered

    if len(aux_tensors) != 8 or aux_tensors[0].ndim != 2:
        raise ValueError("Expected eight THD fixed mask tensors")
    if min(block_size, query_length, key_length, tile_m, tile_n) <= 0:
        raise ValueError("Invalid THD fixed sparse geometry")
    anchors = aux_tensors[0]
    batch = anchors.shape[0]
    for index, tensor in enumerate(aux_tensors):
        shape = (batch, query_length if index < 4 else key_length)
        if (
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != anchors.device
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError("Expected contiguous int32 THD aux tensors on one CUDA device")
    nm, nn = triton.cdiv(query_length, tile_m), triton.cdiv(key_length, tile_n)
    classes = torch.empty((batch, 1, nm, nn), dtype=torch.int32, device=anchors.device)
    _classify_tiles[(batch, nm, nn)](
        *aux_tensors,
        classes,
        Q=query_length,
        K=key_length,
        BL=block_size,
        TM=tile_m,
        TN=tile_n,
        QM=triton.next_power_of_2(triton.cdiv(tile_m, block_size) + 2),
        KN=triton.next_power_of_2(tile_n),
        NM=nm,
        NN=nn,
        WINDOW=context_window or 0,
        QUERY_WINDOW=query_window or 0,
        CAUSAL=draft_is_causal,
        num_warps=4,
    )
    partial_count, partial_index = _dense_to_ordered(classes == 1)
    full_count, full_index = _dense_to_ordered(classes == 2)
    return BlockSparseTensorsTorch(
        mask_block_cnt=partial_count,
        mask_block_idx=partial_index,
        full_block_cnt=full_count,
        full_block_idx=full_index,
        block_size=(tile_m, tile_n),
    )


def build_thd_sparse_pair(
    *, aux_tensors, block_size, query_length, key_length, query_head_dim, value_head_dim,
    context_window=None, query_window=None, draft_is_causal=False,
):
    """Honor stock FA4's independent forward/backward tile dispatch."""
    from aspec.models.ops.fa4_sparse import (
        FA4SparsePair,
        _fa4_sparse_block_sizes,
        _transpose_fixed_sparse,
    )

    forward_size, backward_size = _fa4_sparse_block_sizes(
        aux_tensors[0].device,
        query_head_dim,
        value_head_dim,
    )
    common = dict(
        aux_tensors=aux_tensors,
        block_size=block_size,
        query_length=query_length,
        key_length=key_length,
        context_window=context_window,
        query_window=query_window, draft_is_causal=draft_is_causal,
    )
    forward = _build_sparse_source(**common, tile_m=forward_size[0], tile_n=forward_size[1])
    backward_source = (
        forward
        if backward_size == forward_size
        else _build_sparse_source(
            **common,
            tile_m=backward_size[0],
            tile_n=backward_size[1],
        )
    )
    return FA4SparsePair(
        forward=forward,
        backward=_transpose_fixed_sparse(backward_source, backward_size),
    )
