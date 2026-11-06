# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted for ASpec from Megatron-LM's DSpark FA4 sparse metadata helpers.
# Redistribution is governed by licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.

"""FA4 block-sparse metadata helpers for Direct CP attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

_SM100_FORWARD_BLOCK_SIZE = (256, 128)
# flash-attn-4 4.0.0b27 runs the SM100 backward kernel with a 128x128
# base tile and a Q subtile factor of two. Sparse metadata therefore covers
# 256 query rows but must retain the kernel's 128-column KV tile.
_SM100_BACKWARD_BLOCK_SIZE = (256, 128)
# Match Megatron-inner fixed mode. FA4 resolves the KV tile from the head
# dimensions below; keep the same 256-row sparse Q tile in both directions.
_SM90_BLOCK_SIZE = (256, 128)


def fixed_mask_aux(tensor: Tensor) -> Tensor:
    """Give singleton columns an unambiguous stride-1 axis for FA4 b34."""
    tensor = tensor.to(torch.int32).contiguous()
    if tensor.ndim == 2 and tensor.shape[1] == 1 and tensor.stride(0) == 1:
        # One spare integer per batch row; values and logical shape stay exact.
        # A contiguous [B,1] (including [1,1]) has two stride-1 axes.
        padded = tensor.new_empty(tensor.shape[0], 2)
        padded[:, :1].copy_(tensor)
        return padded[:, :1]
    return tensor


@dataclass(frozen=True)
class FA4SparsePair:
    forward: object
    backward: object


def _fa4_sparse_block_sizes(
    device: torch.device,
    query_head_dim: int,
    value_head_dim: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return sparse metadata tiles, not the kernel's internal CTA tile sizes."""

    major = torch.cuda.get_device_capability(device)[0]
    if major >= 10:
        return _SM100_FORWARD_BLOCK_SIZE, _SM100_BACKWARD_BLOCK_SIZE
    if major == 9:
        from flash_attn.cute.interface import _tile_size_bwd_sm90, _tile_size_fwd_sm90

        fwd = _tile_size_fwd_sm90(
            query_head_dim,
            value_head_dim,
            False,
            False,
            sparse_block_size_q=_SM90_BLOCK_SIZE[0],
        )
        bwd = _tile_size_bwd_sm90(
            query_head_dim,
            value_head_dim,
            False,
            False,
            sparse_block_size_q=_SM90_BLOCK_SIZE[0],
        )
        return (_SM90_BLOCK_SIZE[0], fwd.n_block_size), (_SM90_BLOCK_SIZE[0], bwd.n_block_size)
    return _SM90_BLOCK_SIZE, _SM90_BLOCK_SIZE


def _ordered_to_dense(counts: Tensor, indices: Tensor, columns: int) -> Tensor:
    slots = torch.arange(indices.shape[-1], device=indices.device)
    valid = slots.view(*((1,) * (indices.ndim - 1)), -1) < counts.unsqueeze(-1)
    dense = torch.zeros((*indices.shape[:-1], columns + 1), dtype=torch.bool, device=indices.device)
    safe = torch.where(valid, indices.to(torch.long), columns)
    dense.scatter_(-1, safe, valid)
    return dense[..., :columns]


def _dense_to_ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
    columns = blocks.shape[-1]
    indices = torch.arange(columns, dtype=torch.int32, device=blocks.device)
    indices = indices.view(*((1,) * (blocks.ndim - 1)), columns).expand_as(blocks)
    sentinel = indices + columns
    packed = torch.sort(torch.where(blocks, indices, sentinel), dim=-1).values
    packed = torch.where(packed < columns, packed, torch.zeros_like(packed))
    return blocks.sum(dim=-1, dtype=torch.int32).contiguous(), packed.contiguous()


def _transpose_fixed_sparse(forward, block_size):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    if forward.mask_block_idx.ndim != 4:
        raise ValueError("fixed sparse metadata must have four-dimensional indices")
    num_n_blocks = forward.mask_block_idx.shape[-1]
    partial = _ordered_to_dense(forward.mask_block_cnt, forward.mask_block_idx, num_n_blocks)
    partial_cnt, partial_idx = _dense_to_ordered(partial.transpose(-1, -2))
    if forward.full_block_cnt is None:
        full_cnt = full_idx = None
    else:
        full = _ordered_to_dense(forward.full_block_cnt, forward.full_block_idx, num_n_blocks)
        full_cnt, full_idx = _dense_to_ordered(full.transpose(-1, -2))
    return BlockSparseTensorsTorch(
        mask_block_cnt=partial_cnt,
        mask_block_idx=partial_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=block_size,
    )


def _build_fixed_sparse(
    *,
    mask_mod: Callable,
    aux_tensors: list[Tensor],
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    block_size: tuple[int, int],
):
    from flash_attn.cute.compute_block_sparsity import compute_block_sparsity

    return compute_block_sparsity(
        tile_m=block_size[0],
        tile_n=block_size[1],
        batch_size=batch_size,
        num_heads=1,
        seqlen_q=query_length,
        seqlen_k=key_length,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        device=device,
    )


def build_fixed_sparse_pair(
    *,
    mask_mod: Callable,
    aux_tensors: list[Tensor],
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    query_head_dim: int,
    value_head_dim: int,
) -> FA4SparsePair:
    """Build fixed forward and transposed backward sparse traversals."""

    common = dict(
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        device=device,
    )
    forward_block_size, backward_block_size = _fa4_sparse_block_sizes(
        device, query_head_dim, value_head_dim
    )
    forward = _build_fixed_sparse(**common, block_size=forward_block_size)
    backward_source = _build_fixed_sparse(**common, block_size=backward_block_size)
    return FA4SparsePair(
        forward=forward,
        backward=_transpose_fixed_sparse(backward_source, backward_block_size),
    )


def _csa_prefix_sparse(
    *,
    anchor_positions: Tensor,
    query_blocks: Tensor,
    compressed_ends: Tensor,
    query_token_keep: Tensor,
    compressed_valid: Tensor,
    block_size: tuple[int, int],
    anchor_doc_ids: Tensor | None = None,
    compressed_doc_ids: Tensor | None = None,
):
    """Analytic tile cover for CSA's compressed causal-prefix branch."""

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    tile_m, tile_n = block_size
    batch, query_length = query_blocks.shape
    key_length = compressed_ends.shape[1]
    if query_token_keep.shape != (batch, query_length):
        raise ValueError("CSA query_token_keep does not match query blocks")
    if compressed_valid.shape != (batch, key_length):
        raise ValueError("CSA compressed validity does not match compressed ends")
    if (anchor_doc_ids is None) != (compressed_doc_ids is None):
        raise ValueError("CSA anchor/compressed documents must be provided together")

    large = torch.iinfo(torch.int32).max
    query_anchor = torch.gather(anchor_positions, 1, query_blocks.long())
    query_anchor = _pad_last(query_anchor, tile_m, -1).unflatten(-1, (-1, tile_m))
    query_valid = _pad_last(query_token_keep.bool(), tile_m, False).unflatten(
        -1, (-1, tile_m)
    )
    query_min = torch.where(query_valid, query_anchor, large).amin(-1)
    query_max = torch.where(query_valid, query_anchor, -1).amax(-1)
    query_all = query_valid.all(-1)

    key_ends = _pad_last(compressed_ends, tile_n, 0).unflatten(
        -1, (-1, tile_n)
    )
    key_valid = _pad_last(compressed_valid.bool(), tile_n, False).unflatten(
        -1, (-1, tile_n)
    )
    key_min = torch.where(key_valid, key_ends, large).amin(-1)
    key_max = torch.where(key_valid, key_ends, -1).amax(-1)
    key_any = key_valid.any(-1)
    key_all = key_valid.all(-1)

    visible = (
        query_valid.any(-1)[:, :, None]
        & key_any[:, None, :]
        & (key_min[:, None, :] <= query_max[:, :, None])
    )
    full = (
        query_all[:, :, None]
        & key_all[:, None, :]
        & (key_max[:, None, :] <= query_min[:, :, None])
    )
    if anchor_doc_ids is not None:
        query_docs = torch.gather(anchor_doc_ids, 1, query_blocks.long())
        query_docs = _pad_last(query_docs, tile_m, -1).unflatten(
            -1, (-1, tile_m)
        )
        query_doc_min = torch.where(query_valid, query_docs, large).amin(-1)
        query_doc_max = torch.where(query_valid, query_docs, -1).amax(-1)
        key_docs = _pad_last(compressed_doc_ids, tile_n, -1).unflatten(
            -1, (-1, tile_n)
        )
        key_doc_min = torch.where(key_valid, key_docs, large).amin(-1)
        key_doc_max = torch.where(key_valid, key_docs, -1).amax(-1)
        may_share_document = (
            query_doc_min[:, :, None] <= key_doc_max[:, None, :]
        ) & (
            key_doc_min[:, None, :] <= query_doc_max[:, :, None]
        )
        visible &= may_share_document
        full &= (
            query_doc_min.eq(query_doc_max)[:, :, None]
            & key_doc_min.eq(key_doc_max)[:, None, :]
            & query_doc_min[:, :, None].eq(key_doc_min[:, None, :])
            & query_doc_min[:, :, None].ge(0)
        )
    partial = visible & ~full
    partial_count, partial_index = _dense_to_ordered(partial.unsqueeze(1))
    full_count, full_index = _dense_to_ordered(full.unsqueeze(1))
    return BlockSparseTensorsTorch(
        mask_block_cnt=partial_count,
        mask_block_idx=partial_index,
        full_block_cnt=full_count,
        full_block_idx=full_index,
        block_size=block_size,
    )


def build_csa_prefix_sparse_pair(
    *,
    anchor_positions: Tensor,
    query_blocks: Tensor,
    compressed_ends: Tensor,
    query_token_keep: Tensor,
    compressed_valid: Tensor,
    query_head_dim: int,
    value_head_dim: int,
    anchor_doc_ids: Tensor | None = None,
    compressed_doc_ids: Tensor | None = None,
) -> FA4SparsePair:
    """Build cached forward/backward sparse traversals for CSA prefixes."""

    common = dict(
        anchor_positions=anchor_positions,
        query_blocks=query_blocks,
        compressed_ends=compressed_ends,
        query_token_keep=query_token_keep,
        compressed_valid=compressed_valid,
        anchor_doc_ids=anchor_doc_ids,
        compressed_doc_ids=compressed_doc_ids,
    )
    forward_block_size, backward_block_size = _fa4_sparse_block_sizes(
        compressed_ends.device,
        query_head_dim,
        value_head_dim,
    )
    forward = _csa_prefix_sparse(**common, block_size=forward_block_size)
    backward_source = _csa_prefix_sparse(**common, block_size=backward_block_size)
    return FA4SparsePair(
        forward=forward,
        backward=_transpose_fixed_sparse(backward_source, backward_block_size),
    )


def _pad_last(tensor: Tensor, multiple: int, value: int | bool) -> Tensor:
    remainder = tensor.shape[-1] % multiple
    if remainder == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, multiple - remainder), value=value)


def _direct_cp_tile_masks(
    *,
    anchor_positions: Tensor,
    query_blocks: Tensor,
    context_positions: Tensor,
    draft_blocks: Tensor,
    query_token_keep: Tensor,
    key_token_keep: Tensor,
    tile_m: int,
    tile_n: int,
    anchor_doc_ids: Tensor | None = None,
    context_doc_ids: Tensor | None = None,
    context_window: int | None = None,
    query_window: int | None = None,
    draft_block_size: int = 1,
) -> tuple[Tensor, Tensor]:
    """Build a conservative exact/sparse tile cover without compiling mask code."""

    batch, query_length = query_blocks.shape
    context_length = context_positions.shape[1]
    key_length = context_length + draft_blocks.shape[1]
    if query_token_keep.shape != (batch, query_length):
        raise ValueError("query_token_keep shape does not match query_blocks")
    if key_token_keep.shape != (batch, key_length):
        raise ValueError("key_token_keep shape does not match Direct CP keys")
    if (anchor_doc_ids is None) != (context_doc_ids is None):
        raise ValueError("anchor_doc_ids and context_doc_ids must be provided together")
    packed = anchor_doc_ids is not None
    if packed:
        if anchor_doc_ids.shape != anchor_positions.shape:
            raise ValueError("anchor_doc_ids must match anchor_positions")
        if context_doc_ids.shape != context_positions.shape:
            raise ValueError("context_doc_ids must match context_positions")

    query_valid = query_token_keep.bool()
    query_anchor = torch.gather(anchor_positions, 1, query_blocks.long())
    query_anchor = _pad_last(query_anchor, tile_m, -1).unflatten(-1, (-1, tile_m))
    query_valid_tiles = _pad_last(query_valid, tile_m, False).unflatten(-1, (-1, tile_m))
    query_blocks_tiles = _pad_last(query_blocks, tile_m, -1).unflatten(-1, (-1, tile_m))
    large = torch.iinfo(torch.int32).max
    query_anchor_min = torch.where(query_valid_tiles, query_anchor, large).amin(-1)
    query_anchor_max = torch.where(query_valid_tiles, query_anchor, -1).amax(-1)
    query_block_min = torch.where(query_valid_tiles, query_blocks_tiles, large).amin(-1)
    query_block_max = torch.where(query_valid_tiles, query_blocks_tiles, -1).amax(-1)
    query_all_valid = query_valid_tiles.all(-1)
    if packed:
        query_docs = torch.gather(anchor_doc_ids, 1, query_blocks.long())
        query_doc_tiles = _pad_last(query_docs, tile_m, -1).unflatten(-1, (-1, tile_m))
        query_doc_min = torch.where(query_valid_tiles, query_doc_tiles, large).amin(-1)
        query_doc_max = torch.where(query_valid_tiles, query_doc_tiles, -1).amax(-1)

    context_flag = torch.cat(
        (
            torch.ones_like(context_positions, dtype=torch.bool),
            torch.zeros_like(draft_blocks, dtype=torch.bool),
        ),
        dim=1,
    )
    key_context_positions = torch.cat((context_positions, torch.zeros_like(draft_blocks)), dim=1)
    key_draft_blocks = torch.cat((torch.full_like(context_positions, -1), draft_blocks), dim=1)
    key_valid = key_token_keep.bool()

    context_tiles = _pad_last(context_flag, tile_n, False).unflatten(-1, (-1, tile_n))
    key_valid_tiles = _pad_last(key_valid, tile_n, False).unflatten(-1, (-1, tile_n))
    context_valid = context_tiles & key_valid_tiles
    context_position_tiles = _pad_last(key_context_positions, tile_n, 0).unflatten(-1, (-1, tile_n))
    context_min = torch.where(context_valid, context_position_tiles, large).amin(-1)
    context_max = torch.where(context_valid, context_position_tiles, -1).amax(-1)
    context_any = context_valid.any(-1)
    context_full_tile = context_valid.all(-1)
    if packed:
        key_context_docs = torch.cat((context_doc_ids, torch.full_like(draft_blocks, -1)), dim=1)
        context_doc_tiles = _pad_last(key_context_docs, tile_n, -1).unflatten(-1, (-1, tile_n))
        context_doc_min = torch.where(context_valid, context_doc_tiles, large).amin(-1)
        context_doc_max = torch.where(context_valid, context_doc_tiles, -1).amax(-1)

    draft_valid = (~context_tiles) & key_valid_tiles
    draft_block_tiles = _pad_last(key_draft_blocks, tile_n, -1).unflatten(-1, (-1, tile_n))
    draft_min = torch.where(draft_valid, draft_block_tiles, large).amin(-1)
    draft_max = torch.where(draft_valid, draft_block_tiles, -1).amax(-1)
    draft_any = draft_valid.any(-1)

    visible_context = context_any[:, None, :] & (
        query_anchor_max[:, :, None] > context_min[:, None, :]
    )
    if context_window is not None:
        visible_context &= context_max[:, None, :] >= query_anchor_min[:, :, None] - context_window
    visible_draft = (
        draft_any[:, None, :]
        & (query_block_min[:, :, None] <= draft_max[:, None, :])
        & (draft_min[:, None, :] <= query_block_max[:, :, None])
    )
    full = (
        query_all_valid[:, :, None]
        & context_full_tile[:, None, :]
        & (query_anchor_min[:, :, None] > context_max[:, None, :])
    )
    if context_window is not None:
        full &= context_min[:, None, :] >= query_anchor_max[:, :, None] - context_window
    if query_window is not None:
        # The cover may include boundary tiles; never mark them full when any
        # query-offset window would need the elementwise mask.
        full &= context_min[:, None, :] > (
            query_anchor_max[:, :, None] + draft_block_size - 1 - query_window
        )
    if packed:
        # FA4 bypasses mask_mod for a full tile. It is safe only when every
        # query and context token belongs to one identical, valid document.
        query_one_doc = (query_doc_min == query_doc_max) & (query_doc_min >= 0)
        context_one_doc = (context_doc_min == context_doc_max) & (context_doc_min >= 0)
        full &= (
            query_one_doc[:, :, None]
            & context_one_doc[:, None, :]
            & (query_doc_min[:, :, None] == context_doc_min[:, None, :])
        )
    partial = (visible_context | visible_draft) & ~full
    return partial.unsqueeze(1), full.unsqueeze(1)


def _direct_cp_sparse(
    *,
    anchor_positions: Tensor,
    query_blocks: Tensor,
    context_positions: Tensor,
    draft_blocks: Tensor,
    query_token_keep: Tensor,
    key_token_keep: Tensor,
    block_size: tuple[int, int],
    anchor_doc_ids: Tensor | None = None,
    context_doc_ids: Tensor | None = None,
    context_window: int | None = None,
    query_window: int | None = None,
    draft_block_size: int = 1,
):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    partial, full = _direct_cp_tile_masks(
        anchor_positions=anchor_positions,
        query_blocks=query_blocks,
        context_positions=context_positions,
        draft_blocks=draft_blocks,
        query_token_keep=query_token_keep,
        key_token_keep=key_token_keep,
        tile_m=block_size[0],
        tile_n=block_size[1],
        anchor_doc_ids=anchor_doc_ids,
        context_doc_ids=context_doc_ids,
        context_window=context_window,
        query_window=query_window,
        draft_block_size=draft_block_size,
    )
    partial_cnt, partial_idx = _dense_to_ordered(partial)
    full_cnt, full_idx = _dense_to_ordered(full)
    return BlockSparseTensorsTorch(
        mask_block_cnt=partial_cnt,
        mask_block_idx=partial_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=block_size,
    )


def build_direct_cp_sparse_pair(
    *,
    anchor_positions: Tensor,
    query_blocks: Tensor,
    context_positions: Tensor,
    draft_blocks: Tensor,
    query_token_keep: Tensor,
    key_token_keep: Tensor,
    query_head_dim: int,
    value_head_dim: int,
    anchor_doc_ids: Tensor | None = None,
    context_doc_ids: Tensor | None = None,
    context_window: int | None = None,
    query_window: int | None = None,
    draft_block_size: int = 1,
) -> FA4SparsePair:
    """Build DSpark sparse traversals directly from its causal/block structure."""

    common = dict(
        anchor_positions=anchor_positions,
        query_blocks=query_blocks,
        context_positions=context_positions,
        draft_blocks=draft_blocks,
        query_token_keep=query_token_keep,
        key_token_keep=key_token_keep,
        anchor_doc_ids=anchor_doc_ids,
        context_doc_ids=context_doc_ids,
        context_window=context_window,
        query_window=query_window,
        draft_block_size=draft_block_size,
    )
    forward_block_size, backward_block_size = _fa4_sparse_block_sizes(
        anchor_positions.device, query_head_dim, value_head_dim
    )
    forward = _direct_cp_sparse(**common, block_size=forward_block_size)
    backward_source = _direct_cp_sparse(**common, block_size=backward_block_size)
    return FA4SparsePair(
        forward=forward,
        backward=_transpose_fixed_sparse(backward_source, backward_block_size),
    )


__all__ = ["FA4SparsePair", "build_direct_cp_sparse_pair", "build_fixed_sparse_pair"]
