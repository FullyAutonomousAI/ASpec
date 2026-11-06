# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from the NVIDIA Megatron-LM DSpark packed-attention implementation (fixed mode).
# Redistribution is governed by licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.

"""Unpadded THD packing through FA4's fixed sparse forward/backward API.

One physical packed row contains concatenated documents, not a padded batch of
documents. The singleton batch dimension is only the fixed API adapter. Runtime
document IDs isolate attention; physical row positions and RoPE remain separate.
No patched FA4, varlen API or private backward is used.
"""

from functools import lru_cache

import torch
from torch import Tensor

from aspec.models.ops.direct_cp import (
    MASK_VALUE,
    direct_cp_attention_observation,
)


def _query_metadata(metadata, device, blocks=None):
    blocks = metadata.collective_block_ids.to(device) if blocks is None else blocks.to(device)
    anchors = metadata.anchor_positions[0].index_select(0, blocks).to(device)
    docs = metadata.anchor_doc_ids[0].index_select(0, blocks).to(device)
    keep = metadata.block_keep_mask[0].index_select(0, blocks).to(device)
    offsets = torch.arange(metadata.block_size, device=device).repeat(blocks.numel())
    token_keep = keep.repeat_interleave(metadata.block_size) & (
        anchors.repeat_interleave(metadata.block_size) + offsets < metadata.sequence_length
    ) & (docs.repeat_interleave(metadata.block_size) >= 0)
    return (
        docs.repeat_interleave(metadata.block_size),
        blocks.repeat_interleave(metadata.block_size),
        anchors.repeat_interleave(metadata.block_size),
        token_keep,
    )


def _key_metadata(metadata, device):
    query_docs, query_blocks, _, query_keep = _query_metadata(
        metadata, device, metadata.local_block_ids
    )
    context_docs = metadata.context_doc_ids[0].to(device)
    context_positions = metadata.context_position_ids[0].to(device)
    return (
        torch.cat((context_docs, query_docs)),
        # Negative values encode Draft offsets without specializing a mask on
        # the packed context length. Context RoPE/physical positions stay >= 0.
        torch.cat((context_positions, -(torch.arange(query_docs.numel(), device=device)
                                        % metadata.block_size) - 1)),
        torch.cat((torch.full_like(context_docs, -1), query_blocks)),
        torch.cat((context_docs >= 0, query_keep)),
    )


@lru_cache(maxsize=32)
def _build_fixed_mask(context_window=None, query_window=None, draft_is_causal=False,
                      block_size=1):
    """Build one runtime-shape mask shared by all flattened fixed shapes."""

    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def read_aux(aux, batch, index: "cute.TensorSSA", max_index):
        index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        index_fragment.store(index)
        value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        value_fragment[0] = aux[
            batch[0], cutlass.min(index_fragment[0], max_index - 1)
        ]
        return value_fragment.load()

    @cute.jit
    def mask_mod(batch, head, query_index, key_index, seqlen_info, aux_tensors):
        del head
        (
            query_anchors,
            query_keep,
            query_blocks,
            query_samples,
            key_samples,
            key_positions,
            key_blocks,
            key_keep,
        ) = aux_tensors
        zero = cutlass.Int32(0)
        query_block = read_aux(
            query_blocks, batch, query_index, seqlen_info.seqlen_q
        )
        query_sample = read_aux(
            query_samples, batch, query_index, seqlen_info.seqlen_q
        )
        key_sample = read_aux(
            key_samples, batch, key_index, seqlen_info.seqlen_k
        )
        anchor = read_aux(
            query_anchors, batch, query_index, seqlen_info.seqlen_q
        )
        keep = read_aux(query_keep, batch, query_index, seqlen_info.seqlen_q)
        context_position = read_aux(
            key_positions, batch, key_index, seqlen_info.seqlen_k
        )
        draft_block = read_aux(
            key_blocks, batch, key_index, seqlen_info.seqlen_k
        )
        visible_key = read_aux(
            key_keep, batch, key_index, seqlen_info.seqlen_k
        )
        visible_context = (context_position >= zero) & (context_position < anchor)
        if cutlass.const_expr(context_window is not None):
            visible_context = visible_context & (context_position >= anchor - cutlass.Int32(context_window))
        visible_draft = (draft_block >= zero) & (draft_block == query_block)
        query_offset = query_index % block_size
        draft_offset = -context_position - 1
        if cutlass.const_expr(query_window is not None):
            window = cutlass.Int32(query_window)
            visible_context = visible_context & (anchor + query_offset - context_position < window)
            distance = query_offset - draft_offset
            visible_draft = visible_draft & (distance < window) & (distance > -window)
        if cutlass.const_expr(draft_is_causal):
            visible_draft = visible_draft & (draft_offset <= query_offset)
        return (
            (query_sample == key_sample)
            & (visible_context | visible_draft)
            & (keep > zero)
            & (visible_key > zero)
        )

    return mask_mod


def thd_fa4_attention(query: Tensor, key: Tensor, value: Tensor, metadata, scale):
    from flash_attn.cute.interface import flash_attn_func

    from aspec.models.ops.fa4_sparse import fixed_mask_aux

    if query.shape[0] != 1:
        raise ValueError("THD packing requires one controller-owned row per microbatch")
    if metadata.context_doc_ids is None or metadata.sequence_length is None:
        raise ValueError("THD requires document boundaries and physical ownership")
    if query.shape[1] % key.shape[1]:
        raise ValueError("THD query heads must be divisible by KV heads")

    input_key = (query.device, query.shape[2], key.shape[2])
    inputs = metadata._fa4_cache.thd_inputs.get(input_key)
    if inputs is None:
        query_docs, query_blocks, anchors, query_keep = _query_metadata(metadata, query.device)
        key_docs, key_positions, key_blocks, key_keep = _key_metadata(metadata, query.device)
        fixed_aux = [
            fixed_mask_aux(tensor.unsqueeze(0)) for tensor in
            (anchors, query_keep, query_blocks, query_docs,
             key_docs, key_positions, key_blocks, key_keep)
        ]
        # Empty-partition detection and token-major inputs are metadata too;
        # neither needs a per-layer GPU allocation or host synchronization.
        inputs = (fixed_aux, bool(query_keep.any() and key_keep.any()))
        metadata._fa4_cache.thd_inputs[input_key] = inputs
    fixed_aux, has_visible_rows = inputs
    observation = direct_cp_attention_observation(metadata)
    observation.update(
        qkv_format="thd", fa4_mode="fixed", backward_strategy="fa4_fixed_sparse",
        backward_observed=False, context_parallel_size=metadata.cp_size,
        context_parallel_rank=metadata.cp_rank, empty_local_partition=False,
        query_tokens=query.shape[2], key_tokens=key.shape[2],
        context_tokens=metadata.context_length, context_window=metadata.context_window,
        query_window=metadata.query_window, draft_is_causal=metadata.draft_is_causal,
    )
    if not has_visible_rows:
        observation["empty_local_partition"] = True
        zero = query.sum() * 0 + key.sum() * 0 + value.sum() * 0
        output = zero + value.new_zeros(
            query.shape[0], query.shape[1], query.shape[2], value.shape[-1]
        )
        lse = zero + torch.full(
            query.shape[:3], MASK_VALUE, device=query.device, dtype=torch.float32
        )
        return output, lse

    cache_key = ("thd_fixed", query.device, query.shape[2], key.shape[2],
                 query.shape[-1], value.shape[-1], metadata.context_window,
                 metadata.query_window, metadata.draft_is_causal)
    sparse = metadata._fa4_cache.get(cache_key)
    mask_mod = _build_fixed_mask(
        metadata.context_window, metadata.query_window, metadata.draft_is_causal,
        metadata.block_size,
    )
    if metadata.fa4_sparse and sparse is None:
        from aspec.models.ops.thd_sparse_metadata import build_thd_sparse_pair

        sparse = build_thd_sparse_pair(
            aux_tensors=fixed_aux, block_size=metadata.block_size,
            context_window=metadata.context_window,
            query_window=metadata.query_window, draft_is_causal=metadata.draft_is_causal,
            query_length=query.shape[2], key_length=key.shape[2],
            query_head_dim=query.shape[-1], value_head_dim=value.shape[-1],
        )
        metadata._fa4_cache[cache_key] = sparse
    output, lse = flash_attn_func(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        softmax_scale=scale, causal=False, mask_mod=mask_mod,
        aux_tensors=fixed_aux,
        block_sparse_tensors=None if sparse is None else sparse.forward,
        block_sparse_tensors_bwd=None if sparse is None else sparse.backward,
        return_lse=True,
    )[:2]

    def observe_backward(gradient):
        observation["backward_observed"] = True
        return gradient

    # Observation follows the actual kernel tensors even when FSDP replaces
    # metadata kwargs. A successful optimizer commit also requires backward to finish.
    for tensor in (output, lse):
        if tensor.requires_grad:
            tensor.register_hook(observe_backward)
    return output.transpose(1, 2).contiguous(), lse
