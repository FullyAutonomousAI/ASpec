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

"""Bounded selected-key FlashAttention for token-major Draft attention.

The caller supplies one compact key-index row per independent query sequence.
This keeps sparse CSA/QSA-style execution out of padded BSHD and avoids a
query-by-full-context mask or retained gather workspace during backward.
"""

from __future__ import annotations

import torch


def compact_varlen_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_valid: torch.Tensor,
    sequence_keep: torch.Tensor,
    queries_per_sequence: int,
    scale: float,
    workspace_bytes: int,
    return_lse: bool = False,
    observation: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attend each query sequence to its compact, explicitly selected keys."""

    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise NotImplementedError("compact sparse attention requires CUDA FP16/BF16")
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except ImportError as error:
        raise NotImplementedError(
            "compact sparse attention requires FlashAttention-4 varlen support"
        ) from error
    from torch.utils.checkpoint import checkpoint

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("compact sparse attention expects BSHD tensor views")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("compact sparse key/value leading shapes must match")
    batch, query_length, query_heads, query_dim = query.shape
    if key.shape[0] != batch or query_heads % key.shape[2]:
        raise ValueError("query heads must be divisible by compact KV heads")
    if token_indices.ndim != 3 or token_valid.shape != token_indices.shape:
        raise ValueError("compact token indices and validity must be [B, sequences, capacity]")
    if token_indices.shape[:2] != sequence_keep.shape:
        raise ValueError("compact sequence_keep must match token-index rows")
    if queries_per_sequence <= 0 or query_length != token_indices.shape[1] * queries_per_sequence:
        raise ValueError("query rows do not match compact attention sequences")
    if workspace_bytes <= 0:
        raise ValueError("compact sparse workspace must be positive")

    sequence_count = token_indices.shape[1]
    row_has_keys = token_valid.any(dim=-1)
    active_rows = (
        sequence_keep.bool() & row_has_keys
    ).reshape(-1).nonzero(as_tuple=False).flatten()
    if observation is not None:
        observation["empty_local_partition"] = active_rows.numel() == 0
    output_rows = query.new_zeros(
        batch * sequence_count,
        queries_per_sequence,
        query_heads,
        value.shape[-1],
    )
    lse_rows = torch.full(
        (batch * sequence_count, query_heads, queries_per_sequence),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    if active_rows.numel() == 0:
        # Empty owners must still enter the same CP backward collectives.
        zero = query.sum() * 0 + key.sum() * 0 + value.sum() * 0
        output_rows = output_rows + zero
        lse_rows = lse_rows + zero
        output = output_rows.view(batch, query_length, query_heads, value.shape[-1])
        lse = lse_rows.view(batch, sequence_count, query_heads, queries_per_sequence)
        lse = lse.permute(0, 2, 1, 3).flatten(2, 3)
        return (output, lse) if return_lse else output

    query_rows = query.view(
        batch * sequence_count, queries_per_sequence, query_heads, query_dim
    ).index_select(0, active_rows)
    index_rows = token_indices.view(batch * sequence_count, -1).index_select(
        0, active_rows
    )
    valid_rows = token_valid.view(batch * sequence_count, -1).index_select(
        0, active_rows
    )
    row_batches = (
        torch.arange(batch, device=query.device)
        .view(-1, 1)
        .expand(-1, sequence_count)
        .reshape(-1)
        .index_select(0, active_rows)
    )
    selected_capacity = index_rows.shape[1]
    bytes_per_row = max(
        selected_capacity
        * key.shape[2]
        * (key.shape[-1] * key.element_size() + value.shape[-1] * value.element_size()),
        1,
    )
    rows_per_chunk = max(1, workspace_bytes // bytes_per_row)
    output_chunks = []
    lse_chunks = []
    for start in range(0, active_rows.numel(), rows_per_chunk):
        end = min(start + rows_per_chunk, active_rows.numel())
        q_rows = query_rows[start:end]
        index_chunk = index_rows[start:end]
        valid_chunk = valid_rows[start:end]
        batch_chunk = row_batches[start:end]

        def attend(
            q_input,
            k_input,
            v_input,
            index_input,
            valid_input,
            batch_input,
            _max_key_length=selected_capacity,
        ):
            flat_valid = valid_input.reshape(-1)
            flat_indices = index_input.reshape(-1)[flat_valid]
            flat_batches = (
                batch_input[:, None]
                .expand_as(index_input)
                .reshape(-1)[flat_valid]
            )
            gathered_key = k_input[flat_batches, flat_indices].contiguous()
            gathered_value = v_input[flat_batches, flat_indices].contiguous()
            q_flat = q_input.reshape(-1, query_heads, query_dim).contiguous()
            counts = valid_input.sum(dim=-1, dtype=torch.int32)
            cu_query = torch.arange(
                0,
                (q_input.shape[0] + 1) * queries_per_sequence,
                queries_per_sequence,
                dtype=torch.int32,
                device=q_input.device,
            )
            cu_key = torch.cat(
                (counts.new_zeros(1), counts.cumsum(dim=0, dtype=torch.int32)),
                dim=0,
            )
            output, lse = flash_attn_varlen_func(
                q_flat,
                gathered_key,
                gathered_value,
                cu_seqlens_q=cu_query,
                cu_seqlens_k=cu_key,
                max_seqlen_q=queries_per_sequence,
                max_seqlen_k=_max_key_length,
                softmax_scale=scale,
                causal=False,
                pack_gqa=query_heads != key.shape[2],
                return_lse=True,
            )
            output = output.view(
                q_input.shape[0], queries_per_sequence, query_heads, value.shape[-1]
            )
            lse = lse.transpose(0, 1).reshape(
                q_input.shape[0], queries_per_sequence, query_heads
            ).transpose(1, 2)
            return output, lse

        output_chunk, lse_chunk = checkpoint(
            attend,
            q_rows,
            key,
            value,
            index_chunk,
            valid_chunk,
            batch_chunk,
            use_reentrant=False,
        )
        output_chunks.append(output_chunk)
        lse_chunks.append(lse_chunk)
    active_output = torch.cat(output_chunks, dim=0)
    active_lse = torch.cat(lse_chunks, dim=0)
    output_rows = output_rows.index_copy(0, active_rows, active_output)
    lse_rows = lse_rows.index_copy(0, active_rows, active_lse)
    output = output_rows.view(batch, query_length, query_heads, value.shape[-1])
    lse = lse_rows.view(batch, sequence_count, query_heads, queries_per_sequence)
    lse = lse.permute(0, 2, 1, 3).flatten(2, 3)
    return (output, lse) if return_lse else output


def distributed_compact_varlen_attention(
    *,
    query_own: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_valid: torch.Tensor,
    sequence_keep: torch.Tensor,
    queries_per_sequence: int,
    scale: float,
    workspace_bytes: int,
    group,
    observation: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge bounded local selected-key attention across context owners."""

    import torch.distributed as dist

    from aspec.models.ops.direct_cp import (
        all_gather_query_rows,
        all_to_all_query_partials,
        merge_attention_partials,
    )

    cp_size = dist.get_world_size(group) if group is not None else 1
    query_full = (
        query_own
        if cp_size == 1
        else all_gather_query_rows(query_own, group=group)
    )
    partial_output, partial_lse = compact_varlen_attention(
        query=query_full,
        key=key,
        value=value,
        token_indices=token_indices,
        token_valid=token_valid,
        sequence_keep=sequence_keep,
        queries_per_sequence=queries_per_sequence,
        scale=scale,
        workspace_bytes=workspace_bytes,
        return_lse=True,
        observation=observation,
    )
    if cp_size == 1:
        return partial_output, partial_lse
    output_parts = all_to_all_query_partials(
        partial_output.permute(0, 2, 1, 3), group=group
    )
    lse_parts = all_to_all_query_partials(partial_lse, group=group)
    output, lse = merge_attention_partials(output_parts, lse_parts)
    return output.permute(0, 2, 1, 3).to(query_own.dtype), lse


__all__ = ["compact_varlen_attention", "distributed_compact_varlen_attention"]
