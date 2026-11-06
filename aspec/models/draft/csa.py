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

"""Compressed Sparse Attention for the DSpark/DFlash Draft path.

The compressor and Lightning Indexer follow the public DeepSeek-V4 CSA
semantics, adapted to DFlash's anchor-shared context and GQA projections.  CSA2
cross-layer cache/index reuse is intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import lcm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CSAForwardMetadata:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    block_size: int
    context_window: int
    context_doc_ids: torch.Tensor | None = None
    anchor_doc_ids: torch.Tensor | None = None
    context_physical_position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.anchor_positions.ndim != 2:
            raise ValueError("CSA anchor_positions must be [batch, anchors]")
        if self.block_keep_mask.shape != self.anchor_positions.shape:
            raise ValueError("CSA block_keep_mask must match anchor_positions")
        if self.block_size <= 0 or self.context_window <= 0:
            raise ValueError("CSA block_size and context_window must be positive")
        if (self.context_doc_ids is None) != (self.anchor_doc_ids is None):
            raise ValueError("CSA context_doc_ids and anchor_doc_ids must be provided together")
        if self.context_doc_ids is not None:
            if self.context_doc_ids.ndim != 2:
                raise ValueError("CSA context_doc_ids must be [batch, context]")
            if self.context_doc_ids.shape[0] != self.anchor_positions.shape[0]:
                raise ValueError("CSA context and anchor batches must match")
            if self.anchor_doc_ids.shape != self.anchor_positions.shape:
                raise ValueError("CSA anchor_doc_ids must match anchor_positions")
        if self.context_physical_position_ids is not None:
            if self.context_physical_position_ids.ndim != 2:
                raise ValueError("CSA physical context positions must be [batch, context]")
            if (
                self.context_doc_ids is not None
                and self.context_physical_position_ids.shape
                != self.context_doc_ids.shape
            ):
                raise ValueError("CSA physical positions must match the local context")

    @property
    def query_anchors(self) -> torch.Tensor:
        return self.anchor_positions.repeat_interleave(self.block_size, dim=1)

    @property
    def query_keep(self) -> torch.Tensor:
        return self.block_keep_mask.repeat_interleave(self.block_size, dim=1).bool()

    @property
    def query_docs(self) -> torch.Tensor | None:
        if self.anchor_doc_ids is None:
            return None
        return self.anchor_doc_ids.repeat_interleave(self.block_size, dim=1)


@dataclass(frozen=True)
class CSACompressionLayout:
    token_indices: torch.Tensor
    valid: torch.Tensor
    physical_starts: torch.Tensor
    physical_ends: torch.Tensor
    rope_positions: torch.Tensor
    doc_ids: torch.Tensor

    @property
    def num_groups(self) -> int:
        return self.token_indices.shape[1]


def build_csa_compression_layout(
    metadata: CSAForwardMetadata,
    context_rope_position_ids: torch.Tensor,
    ratio: int,
) -> CSACompressionLayout:
    """Build non-crossing, stride-``ratio`` groups inside each packed document."""

    if ratio < 2:
        raise ValueError("CSA compression ratio must be at least two")
    if context_rope_position_ids.ndim != 2:
        raise ValueError("CSA context RoPE positions must be [batch, context]")
    batch, context_length = context_rope_position_ids.shape
    if batch != metadata.anchor_positions.shape[0]:
        raise ValueError("CSA context and anchor batches must match")
    device = context_rope_position_ids.device
    max_groups = context_length // ratio
    if max_groups == 0:
        shape = (batch, 0)
        return CSACompressionLayout(
            token_indices=torch.empty(batch, 0, ratio, dtype=torch.long, device=device),
            valid=torch.empty(shape, dtype=torch.bool, device=device),
            physical_starts=torch.empty(shape, dtype=torch.long, device=device),
            physical_ends=torch.empty(shape, dtype=torch.long, device=device),
            rope_positions=torch.empty(shape, dtype=torch.long, device=device),
            doc_ids=torch.empty(shape, dtype=torch.long, device=device),
        )

    if metadata.context_doc_ids is None:
        doc_ids = torch.zeros(batch, context_length, dtype=torch.long, device=device)
    else:
        if metadata.context_doc_ids.shape != (batch, context_length):
            raise ValueError("CSA context_doc_ids must match the context sequence")
        doc_ids = metadata.context_doc_ids.to(device=device, dtype=torch.long)

    positions = torch.arange(context_length, device=device, dtype=torch.long).view(1, -1)
    distributed_positions = metadata.context_physical_position_ids is not None
    physical_positions = metadata.context_physical_position_ids
    if not distributed_positions:
        physical_positions = positions.expand(batch, -1)
    elif physical_positions.shape != (batch, context_length):
        raise ValueError("CSA physical positions must match the local context")
    else:
        physical_positions = physical_positions.to(device=device, dtype=torch.long)
    boundary = torch.ones(batch, context_length, dtype=torch.bool, device=device)
    if context_length > 1:
        boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    run_starts = torch.cummax(
        torch.where(boundary, positions, torch.zeros_like(positions)), dim=1
    ).values
    within_run = positions - run_starts
    candidate = (doc_ids >= 0) & (
        context_rope_position_ids.remainder(ratio).eq(0)
        if distributed_positions else within_run.remainder(ratio).eq(0)
    )
    end_positions = positions + ratio - 1
    safe_ends = end_positions.clamp(max=context_length - 1)
    end_docs = torch.gather(doc_ids, 1, safe_ends.expand(batch, -1))
    end_rope = torch.gather(
        context_rope_position_ids, 1, safe_ends.expand(batch, -1)
    )
    candidate &= (end_positions < context_length) & (end_docs == doc_ids)
    if distributed_positions:
        candidate &= end_rope == context_rope_position_ids + ratio - 1

    compact_ids = candidate.to(torch.long).cumsum(dim=1) - 1
    starts = torch.full((batch, max_groups), -1, dtype=torch.long, device=device)
    start_values = positions.expand(batch, -1)
    starts.scatter_reduce_(
        1,
        compact_ids.clamp_min(0),
        torch.where(candidate, start_values, -1),
        reduce="amax",
        include_self=True,
    )
    valid = starts >= 0
    safe_starts = starts.clamp_min(0)
    offsets = torch.arange(ratio, device=device, dtype=torch.long).view(1, 1, -1)
    token_indices = safe_starts.unsqueeze(-1) + offsets
    group_docs = torch.gather(doc_ids, 1, safe_starts)
    rope_positions = torch.gather(context_rope_position_ids, 1, safe_starts)
    physical_starts = torch.gather(physical_positions, 1, safe_starts)
    physical_ends = torch.gather(
        physical_positions,
        1,
        (safe_starts + ratio - 1).clamp_max(context_length - 1),
    ) + 1
    return CSACompressionLayout(
        token_indices=token_indices,
        valid=valid,
        physical_starts=torch.where(valid, physical_starts, torch.zeros_like(starts)),
        physical_ends=torch.where(valid, physical_ends, torch.zeros_like(starts)),
        rope_positions=torch.where(valid, rope_positions, torch.zeros_like(rope_positions)),
        doc_ids=torch.where(valid, group_docs, torch.full_like(group_docs, -1)),
    )


def _csa_remote_predecessor_halo(
    *,
    context_hidden: torch.Tensor,
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    ratio: int,
    direct_cp_metadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fetch only cross-owner predecessor groups used by CSA compression.

    CSA's first series consumes the previous compression group.  Aligned CP
    ownership keeps complete groups local, but the first group of each local
    document shard still needs one predecessor group from its owner.  The
    request plan is integer-only and cached for this forward; activations are
    never cached and the transfer scales with packed documents, not context
    length.
    """

    if direct_cp_metadata is None or direct_cp_metadata.cp_size == 1:
        return None
    cache_key = (
        "csa_predecessor_halo",
        ratio,
        context_hidden.device,
        layout.num_groups,
        metadata.anchor_positions.shape[1],
    )
    plan = direct_cp_metadata._fa4_cache.get(cache_key)
    if plan is None:
        device = context_hidden.device
        batch, anchors = metadata.anchor_positions.shape
        if metadata.anchor_doc_ids is None:
            anchor_docs = torch.zeros(
                batch, anchors, dtype=torch.long, device=device
            )
        else:
            anchor_docs = metadata.anchor_doc_ids.to(device=device, dtype=torch.long)
        anchor_valid = metadata.block_keep_mask.to(device).bool() & anchor_docs.ge(0)
        slots = torch.arange(anchors, device=device)
        earlier_slot = slots.view(1, 1, -1) < slots.view(1, -1, 1)
        has_earlier = (
            anchor_docs[:, :, None].eq(anchor_docs[:, None, :])
            & anchor_valid[:, None, :]
            & earlier_slot
        ).any(dim=-1)
        first_for_document = anchor_valid & ~has_earlier
        active_slots = first_for_document.any(dim=0).nonzero(
            as_tuple=False
        ).flatten()
        requested_docs = anchor_docs.index_select(1, active_slots)

        if active_slots.numel() == 0:
            target_groups = torch.empty(
                batch, 0, dtype=torch.long, device=device
            )
            request_valid = torch.empty(
                batch, 0, dtype=torch.bool, device=device
            )
            requested_positions = torch.empty(
                batch, 0, dtype=torch.long, device=device
            )
        elif layout.num_groups == 0:
            # Requests are all-gathered with equal shapes. An empty owner must
            # still enter the exchange, including its backward collective.
            target_groups = torch.zeros_like(requested_docs)
            request_valid = torch.zeros_like(requested_docs, dtype=torch.bool)
            requested_positions = torch.full(
                (batch, active_slots.numel() * ratio), -1,
                dtype=torch.long, device=device,
            )
        else:
            sentinel = torch.iinfo(torch.long).max
            searchable_docs = torch.where(
                layout.valid,
                layout.doc_ids,
                torch.full_like(layout.doc_ids, sentinel),
            ).contiguous()
            target_groups = torch.searchsorted(
                searchable_docs,
                requested_docs.contiguous(),
                right=False,
            )
            safe_groups = target_groups.clamp_max(layout.num_groups - 1)
            found_docs = torch.gather(layout.doc_ids, 1, safe_groups)
            found = (
                target_groups.lt(layout.num_groups)
                & requested_docs.ge(0)
                & found_docs.eq(requested_docs)
                & first_for_document.index_select(1, active_slots)
            )
            first_starts = torch.gather(layout.physical_starts, 1, safe_groups)
            request_valid = found & first_starts.ge(ratio)
            predecessor_offsets = torch.arange(
                ratio, device=device, dtype=torch.long
            ).view(1, 1, -1)
            requested_positions = (
                first_starts.unsqueeze(-1) - ratio + predecessor_offsets
            )
            requested_positions = torch.where(
                request_valid.unsqueeze(-1),
                requested_positions,
                torch.full_like(requested_positions, -1),
            ).flatten(1)
            target_groups = safe_groups
        plan = (requested_positions, target_groups, request_valid)
        direct_cp_metadata._fa4_cache[cache_key] = plan

    requested_positions, target_groups, request_valid = plan
    if requested_positions.shape[1] == 0:
        return None
    if direct_cp_metadata.sequence_length is None:
        raise ValueError("CSA Direct CP requires the global sequence length")
    from aspec.models.ops.direct_cp import gather_owner_sharded_rows

    remote_hidden = gather_owner_sharded_rows(
        context_hidden,
        direct_cp_metadata.context_position_ids.to(context_hidden.device),
        requested_positions,
        global_sequence_length=direct_cp_metadata.sequence_length,
        group=direct_cp_metadata.group,
    )
    return (
        remote_hidden.view(
            context_hidden.shape[0], target_groups.shape[1], ratio, -1
        ),
        target_groups,
        request_valid,
    )


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        if dim % 2:
            raise ValueError("CSA rotary dimensions must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base = base
        self.frequency_dim = dim

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Rebuild from the scalar definition after dtype/device changes. Casting
        # rounded BF16 frequencies back to FP32 cannot recover the phase.
        self.inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.frequency_dim, 2,
                    device=self.inv_freq.device, dtype=torch.float32,
                )
                / self.frequency_dim
            )
        )
        return self

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # An einsum may be autocast to BF16/FP16 before the trigonometry.
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.float()
        cos = torch.cat((freqs, freqs), dim=-1).cos().unsqueeze(-2).to(x.dtype)
        sin = torch.cat((freqs, freqs), dim=-1).sin().unsqueeze(-2).to(x.dtype)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin


class CSAOverlappingCompressor(nn.Module):
    """DeepSeek-V4's two-series, 2m-wide softmax-gated compressor."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ratio: int,
        rope_theta: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ratio = ratio
        width = num_heads * head_dim
        self.kv_proj = nn.Linear(hidden_size, 2 * width, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 2 * width, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(ratio, 2 * width, dtype=torch.float32))
        self.norm = _RMSNorm(head_dim, norm_eps)
        self.rotary = _RotaryEmbedding(head_dim, rope_theta)

    def keep_projection_fp32(self) -> None:
        self.kv_proj.to(dtype=torch.float32)
        self.gate_proj.to(dtype=torch.float32)
        self.position_bias.data = self.position_bias.data.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout: CSACompressionLayout,
        remote_predecessor: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = hidden_states.shape
        groups = layout.num_groups
        if groups == 0:
            empty = hidden_states.new_zeros(
                batch, 0, self.num_heads, self.head_dim
            )
            # Keep every rank's compressor and halo in the autograd graph:
            # another owner can consume the context rows supplied by this one.
            zero = hidden_states.reshape(-1)[:0].sum()
            for parameter in self.parameters():
                zero = zero + parameter.reshape(-1)[:0].sum().to(zero.dtype)
            if remote_predecessor is not None:
                zero = zero + remote_predecessor[0].reshape(-1)[:0].sum()
            empty = empty + zero
            return empty, empty

        dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        projected = self.kv_proj(hidden_float)
        gates = self.gate_proj(hidden_float)
        # Gather from the physical context dimension directly.  Expanding the
        # source to [B, groups, context, width] looks free in forward but makes
        # gather backward materialize that logical tensor (128 GiB at the 4K
        # Ling shape).  Flattening only the selected group tokens is exact and
        # keeps the gradient source at [B, context, width].
        flat_indices = layout.token_indices.reshape(batch, -1)
        projected = torch.gather(
            projected,
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1]),
        ).view(batch, groups, self.ratio, -1)
        gates = torch.gather(
            gates,
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, gates.shape[-1]),
        ).view(batch, groups, self.ratio, -1)
        gates = gates + self.position_bias.view(1, 1, self.ratio, -1)
        projected = projected.view(
            batch, groups, self.ratio, 2, self.num_heads, self.head_dim
        )
        gates = gates.view(
            batch, groups, self.ratio, 2, self.num_heads, self.head_dim
        )

        previous_values = torch.roll(projected[:, :, :, 0], shifts=1, dims=1)
        previous_gates = torch.roll(gates[:, :, :, 0], shifts=1, dims=1)
        same_document = torch.zeros_like(layout.valid)
        same_document[:, 1:] = (
            layout.valid[:, 1:]
            & layout.valid[:, :-1]
            & (layout.doc_ids[:, 1:] == layout.doc_ids[:, :-1])
        )
        previous_gates = previous_gates.masked_fill(
            ~same_document[:, :, None, None, None], -torch.inf
        )
        if remote_predecessor is not None:
            remote_hidden, target_groups, remote_valid = remote_predecessor
            remote_projected = self.kv_proj(remote_hidden.float()).view(
                batch,
                target_groups.shape[1],
                self.ratio,
                2,
                self.num_heads,
                self.head_dim,
            )[:, :, :, 0]
            remote_gates = self.gate_proj(remote_hidden.float())
            remote_gates = remote_gates + self.position_bias.view(
                1, 1, self.ratio, -1
            )
            remote_gates = remote_gates.view(
                batch,
                target_groups.shape[1],
                self.ratio,
                2,
                self.num_heads,
                self.head_dim,
            )[:, :, :, 0]
            scatter_index = target_groups[:, :, None, None, None].expand_as(
                remote_projected
            )
            active = remote_valid[:, :, None, None, None]
            remote_values_by_group = torch.zeros_like(previous_values).scatter_add(
                1,
                scatter_index,
                torch.where(active, remote_projected, torch.zeros_like(remote_projected)),
            )
            remote_gates_by_group = torch.zeros_like(previous_gates).scatter_add(
                1,
                scatter_index,
                torch.where(active, remote_gates, torch.zeros_like(remote_gates)),
            )
            remote_group = torch.zeros_like(layout.valid, dtype=torch.int32)
            remote_group.scatter_add_(
                1,
                target_groups,
                remote_valid.to(torch.int32),
            )
            remote_group = remote_group.bool()
            previous_values = torch.where(
                remote_group[:, :, None, None, None],
                remote_values_by_group,
                previous_values,
            )
            previous_gates = torch.where(
                remote_group[:, :, None, None, None],
                remote_gates_by_group,
                previous_gates,
            )
        current_values = projected[:, :, :, 1]
        current_gates = gates[:, :, :, 1]
        values = torch.cat((previous_values, current_values), dim=2)
        gate_logits = torch.cat((previous_gates, current_gates), dim=2)
        pooled = (values * gate_logits.softmax(dim=2)).sum(dim=2).to(dtype)
        pooled = self.norm(pooled)
        pooled = torch.where(layout.valid[:, :, None, None], pooled, torch.zeros_like(pooled))
        rotated = self.rotary(pooled, layout.rope_positions)
        return pooled, rotated


def csa_eligible_groups(
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    start: int = 0,
    end: int | None = None,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    end = (
        metadata.query_anchors.shape[1]
        if query_indices is None else query_indices.numel()
    ) if end is None else end
    if query_indices is None:
        anchors = metadata.query_anchors[:, start:end]
        keep = metadata.query_keep[:, start:end]
        query_docs = metadata.query_docs
        if query_docs is not None:
            query_docs = query_docs[:, start:end]
    else:
        rows = query_indices[start:end].to(metadata.anchor_positions.device)
        anchors = metadata.query_anchors.index_select(1, rows)
        keep = metadata.query_keep.index_select(1, rows)
        query_docs = metadata.query_docs
        if query_docs is not None:
            query_docs = query_docs.index_select(1, rows)
    anchors = anchors.to(layout.physical_ends.device)
    visible = layout.valid[:, None, :] & (
        layout.physical_ends[:, None, :] <= anchors[:, :, None]
    )
    if query_docs is not None:
        query_docs = query_docs.to(layout.doc_ids.device)
        visible &= (
            (query_docs[:, :, None] >= 0)
            & (layout.doc_ids[:, None, :] == query_docs[:, :, None])
        )
    return visible & keep.to(visible.device)[:, :, None]


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_rows = mask.any(dim=-1)
    safe = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    safe = torch.where(valid_rows.unsqueeze(-1), safe, torch.zeros_like(safe))
    return F.log_softmax(safe, dim=-1), valid_rows


def _expand_compressed_heads(keys: torch.Tensor, query_heads: int) -> torch.Tensor:
    if query_heads % keys.shape[2]:
        raise ValueError("CSA query heads must be divisible by compressed KV heads")
    repeats = query_heads // keys.shape[2]
    return keys if repeats == 1 else keys.repeat_interleave(repeats, dim=2)


def _main_attention_target(
    query: torch.Tensor,
    compressed_key: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
    *,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Aggregate detached main-attention probabilities over query heads."""

    expanded_key = _expand_compressed_heads(compressed_key, query.shape[2])
    chunks = []
    with torch.no_grad():
        expanded_key = expanded_key.float()
        for start in range(0, query.shape[1], query_chunk_size):
            end = min(start + query_chunk_size, query.shape[1])
            scores = torch.einsum(
                "bqhd,bchd->bhqc", query[:, start:end].float(), expanded_key
            ) * scale
            chunk_visible = visible[:, start:end]
            score_mask = chunk_visible[:, None].expand(-1, query.shape[2], -1, -1)
            log_prob, valid = _masked_log_softmax(scores, score_mask)
            probabilities = log_prob.exp() * score_mask
            target = probabilities.sum(dim=1)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            target = torch.where(valid.any(dim=1).unsqueeze(-1), target, torch.zeros_like(target))
            chunks.append(target)
    return torch.cat(chunks, dim=1) if chunks else query.new_zeros(query.shape[0], 0, visible.shape[-1])


def _gather_last(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp_min(0)
    return torch.gather(values, -1, safe)


def _selected_main_attention_target(
    query: torch.Tensor,
    compressed_key: torch.Tensor,
    selected_indices: torch.Tensor,
    scale: float,
    *,
    query_chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the selected-set KL target without materializing all QxHxKxD keys."""

    expanded_key = _expand_compressed_heads(compressed_key, query.shape[2])
    key_by_head = expanded_key.permute(0, 2, 1, 3)
    targets = []
    valid_rows = []
    with torch.no_grad():
        for start in range(0, query.shape[1], query_chunk_size):
            end = min(start + query_chunk_size, query.shape[1])
            chunk_indices = selected_indices[:, start:end]
            chunk_valid = chunk_indices >= 0
            safe_indices = chunk_indices.clamp_min(0)
            gather = safe_indices[:, :, None, :, None].expand(
                -1, -1, query.shape[2], -1, query.shape[-1]
            )
            selected_key = torch.gather(
                key_by_head[:, None].expand(
                    -1, end - start, -1, -1, -1
                ),
                3,
                gather,
            )
            scores = torch.einsum(
                "bqhd,bqhkd->bhqk",
                query[:, start:end].float(),
                selected_key.float(),
            ) * scale
            head_mask = chunk_valid[:, None].expand(
                -1, query.shape[2], -1, -1
            )
            log_prob, head_valid = _masked_log_softmax(scores, head_mask)
            target = (log_prob.exp() * head_mask).sum(dim=1)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            targets.append(target)
            valid_rows.append(head_valid.any(dim=1))
    if not targets:
        empty = query.new_zeros(
            query.shape[0], 0, selected_indices.shape[-1], dtype=torch.float32
        )
        return empty, selected_indices.new_zeros(
            selected_indices.shape[:2], dtype=torch.bool
        )
    return torch.cat(targets, dim=1), torch.cat(valid_rows, dim=1)


@torch.no_grad()
def _distributed_csa_selected_target(
    *,
    query: torch.Tensor,
    compressed_key: torch.Tensor,
    local_selected: torch.Tensor,
    scale: float,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSA's selected-set teacher from owner-local compressed keys."""

    batch, query_count, selected_count = local_selected.shape
    query_heads = query.shape[2]
    kv_heads = compressed_key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("CSA query heads must be divisible by compressed KV heads")
    repetitions = query_heads // kv_heads
    world_size = dist.get_world_size(group) if group is not None else 1
    valid = local_selected >= 0
    target = query.new_zeros(
        batch, query_count, selected_count, dtype=torch.float32
    )
    if selected_count == 0:
        return target, valid.any(dim=-1)
    for kv_head in range(kv_heads):
        if compressed_key.shape[1]:
            source = compressed_key[:, :, kv_head]
            gathered = torch.gather(
                source[:, None].expand(-1, query_count, -1, -1),
                2,
                local_selected.clamp_min(0).unsqueeze(-1).expand(
                    -1, -1, -1, source.shape[-1]
                ),
            )
        else:
            gathered = query.new_zeros(
                batch, query_count, selected_count, query.shape[-1]
            )
        grouped_query = query.view(
            batch, query_count, kv_heads, repetitions, query.shape[-1]
        )[:, :, kv_head]
        scores = torch.einsum(
            "bqrd,bqkd->bqrk", grouped_query.float(), gathered.float()
        ) * scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        local_max = scores.amax(dim=-1)
        global_max = local_max.clone()
        if world_size > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        safe_max = torch.where(
            torch.isfinite(global_max), global_max, torch.zeros_like(global_max)
        )
        exponent = torch.exp(scores - safe_max.unsqueeze(-1))
        exponent = torch.where(valid.unsqueeze(2), exponent, torch.zeros_like(exponent))
        denominator = exponent.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(denominator, group=group)
        target.add_(
            (exponent / denominator.clamp_min(1e-20).unsqueeze(-1)).sum(dim=2)
        )
    if world_size > 1:
        dist.all_reduce(target, group=group)
    row_sum = target.sum(dim=-1, keepdim=True)
    row_valid = row_sum.squeeze(-1) > 0
    return target / row_sum.clamp_min(1e-12), row_valid


@torch.no_grad()
def _distributed_csa_full_target(
    *,
    query: torch.Tensor,
    compressed_key: torch.Tensor,
    eligible: torch.Tensor,
    scale: float,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owner-local slice of CSA's full compressed-key teacher."""

    batch, query_count, query_heads, head_dim = query.shape
    kv_heads = compressed_key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("CSA query heads must be divisible by compressed KV heads")
    repetitions = query_heads // kv_heads
    world_size = dist.get_world_size(group) if group is not None else 1
    grouped_query = query.view(
        batch, query_count, kv_heads, repetitions, head_dim
    )
    target = query.new_zeros(
        batch, query_count, compressed_key.shape[1], dtype=torch.float32
    )
    for kv_head in range(kv_heads):
        scores = torch.einsum(
            "bqrd,bkd->bqrk",
            grouped_query[:, :, kv_head].float(),
            compressed_key[:, :, kv_head].float(),
        ) * scale
        head_mask = eligible.unsqueeze(2)
        if scores.shape[-1]:
            local_max = scores.masked_fill(~head_mask, -torch.inf).amax(dim=-1)
        else:
            local_max = scores.new_full(scores.shape[:-1], -torch.inf)
        global_max = local_max.clone()
        if world_size > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        valid = torch.isfinite(global_max)
        safe_max = torch.where(valid, global_max, torch.zeros_like(global_max))
        exponent = torch.where(
            head_mask,
            torch.exp(scores - safe_max.unsqueeze(-1)),
            torch.zeros_like(scores),
        )
        denominator = exponent.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(denominator, group=group)
        target.add_(
            (
                exponent
                / denominator.clamp_min(1e-20).unsqueeze(-1)
            ).sum(dim=2)
        )
    row_sum = target.sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(row_sum, group=group)
    row_valid = row_sum > 0
    return target / row_sum.clamp_min(1e-12).unsqueeze(-1), row_valid


class CSALightningIndexer(nn.Module):
    """Learned ReLU scorer with the separate compressed-key path used by CSA."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.shard_alignment = lcm(*(r for r in config.compress_ratios if r > 0))
        ratio = config.compress_ratios[layer_idx]
        self.key_compressor = CSAOverlappingCompressor(
            hidden_size=config.hidden_size,
            num_heads=1,
            head_dim=self.head_dim,
            ratio=ratio,
            rope_theta=config.compress_rope_theta,
            norm_eps=config.rms_norm_eps,
        )
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary = _RotaryEmbedding(self.head_dim, config.compress_rope_theta)
        self.scale = self.head_dim**-0.5 * self.num_heads**-0.5

    def keep_projection_fp32(self) -> None:
        self.key_compressor.keep_projection_fp32()

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        layout: CSACompressionLayout,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepSeek DSA/CSA optimizes the indexer only through its KL objective.
        draft_input = draft_hidden.detach()
        context_input = context_hidden.detach()
        _, index_key = self.key_compressor(context_input, layout)
        index_key = index_key.squeeze(2)
        query = self.q_proj(draft_input).view(
            *draft_input.shape[:2], self.num_heads, self.head_dim
        )
        query = self.rotary(query, draft_position_ids)
        weights = self.weights_proj(draft_input).float() * self.scale
        scores = torch.einsum(
            "bqhd,bcd->bqhc", query.float(), index_key.float()
        ).relu_()
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        scores = scores.masked_fill(~eligible, torch.finfo(scores.dtype).min)
        topk = min(self.topk, scores.shape[-1])
        if topk == 0:
            indices = torch.empty(
                *scores.shape[:2], 0, dtype=torch.long, device=scores.device
            )
        else:
            indices = scores.topk(topk, dim=-1, sorted=False).indices
            selected_valid = torch.gather(eligible, -1, indices)
            indices = torch.where(selected_valid, indices, torch.full_like(indices, -1))
        return scores, indices

    def forward_distributed_sparse(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        layout: CSACompressionLayout,
        metadata: CSAForwardMetadata,
        main_query: torch.Tensor,
        main_compressed_key: torch.Tensor,
        scale: float,
        workspace_bytes: int,
        direct_cp_metadata,
        remote_predecessor: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select globally while retaining every compressed key on its owner."""

        from aspec.models.ops.direct_cp import all_gather_query_rows
        from aspec.models.ops.distributed_sparse import distributed_topk

        draft_input = draft_hidden.detach()
        detached_predecessor = (
            None
            if remote_predecessor is None
            else (
                remote_predecessor[0].detach(),
                remote_predecessor[1],
                remote_predecessor[2],
            )
        )
        _, index_key = self.key_compressor(
            context_hidden.detach(),
            layout,
            remote_predecessor=detached_predecessor,
        )
        index_key = index_key.squeeze(2)
        query = self.q_proj(draft_input).view(
            *draft_input.shape[:2], self.num_heads, self.head_dim
        )
        query = self.rotary(query, draft_position_ids)
        weights = self.weights_proj(draft_input).float() * self.scale
        if direct_cp_metadata.cp_size > 1:
            query = all_gather_query_rows(query, group=direct_cp_metadata.group)
            weights = all_gather_query_rows(
                weights.unsqueeze(-1), group=direct_cp_metadata.group
            ).squeeze(-1)
            main_query = all_gather_query_rows(
                main_query, group=direct_cp_metadata.group
            )
        local_query_count = draft_hidden.shape[1]
        query_indices = direct_cp_metadata.collective_block_ids.to(query.device)
        query_indices = (
            query_indices[:, None] * metadata.block_size
            + torch.arange(metadata.block_size, device=query.device)[None]
        ).flatten()
        from aspec.models.ops.distributed_sparse import max_context_shard_length

        context_capacity = max_context_shard_length(
            direct_cp_metadata, context_hidden.shape[1], alignment=self.shard_alignment
        )
        group_capacity = context_capacity // self.key_compressor.ratio
        bytes_per_query = max(
            query.shape[0]
            * max(group_capacity, 1)
            * torch.float32.itemsize
            * 2,
            1,
        )
        chunk_size = max(1, workspace_bytes // bytes_per_query)
        selections = []
        losses = []
        for start in range(0, query.shape[1], chunk_size):
            end = min(start + chunk_size, query.shape[1])
            eligible = csa_eligible_groups(
                metadata,
                layout,
                start,
                end,
                query_indices=query_indices,
            )
            scores = torch.einsum(
                "bqhd,bcd->bqhc",
                query[:, start:end].float(),
                index_key.float(),
            ).relu_()
            scores = (scores * weights[:, start:end].unsqueeze(-1)).sum(dim=2)
            winners = distributed_topk(
                scores,
                eligible,
                topk=self.topk,
                group=direct_cp_metadata.group,
            )
            selections.append(winners.local_winner_indices)
            target, main_valid = _distributed_csa_selected_target(
                query=main_query[:, start:end].detach(),
                compressed_key=main_compressed_key.detach(),
                local_selected=winners.local_winner_indices,
                scale=scale,
                group=direct_cp_metadata.group,
            )
            log_prob, valid = _masked_log_softmax(winners.scores, winners.valid)
            loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
            losses.append(torch.where(valid & main_valid, loss, torch.zeros_like(loss)))
        selected = torch.cat(selections, dim=1)
        loss = torch.cat(losses, dim=1)
        rank = dist.get_rank(direct_cp_metadata.group)
        local_start = rank * local_query_count
        return selected, loss[:, local_start : local_start + local_query_count]

    def forward_distributed_warmup(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        layout: CSACompressionLayout,
        metadata: CSAForwardMetadata,
        main_query: torch.Tensor,
        main_compressed_key: torch.Tensor,
        scale: float,
        workspace_bytes: int,
        direct_cp_metadata,
        remote_predecessor: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None,
    ) -> torch.Tensor:
        """Full CSA indexer KL with owner-local compressed keys."""

        from aspec.models.ops.direct_cp import all_gather_query_rows
        from aspec.models.ops.distributed_sparse import distributed_log_softmax

        draft_input = draft_hidden.detach()
        detached_predecessor = (
            None
            if remote_predecessor is None
            else (
                remote_predecessor[0].detach(),
                remote_predecessor[1],
                remote_predecessor[2],
            )
        )
        _, index_key = self.key_compressor(
            context_hidden.detach(),
            layout,
            remote_predecessor=detached_predecessor,
        )
        index_key = index_key.squeeze(2)
        query = self.q_proj(draft_input).view(
            *draft_input.shape[:2], self.num_heads, self.head_dim
        )
        query = self.rotary(query, draft_position_ids)
        weights = self.weights_proj(draft_input).float() * self.scale
        if direct_cp_metadata.cp_size > 1:
            query = all_gather_query_rows(query, group=direct_cp_metadata.group)
            weights = all_gather_query_rows(
                weights.unsqueeze(-1), group=direct_cp_metadata.group
            ).squeeze(-1)
            main_query = all_gather_query_rows(
                main_query, group=direct_cp_metadata.group
            )
        local_query_count = draft_hidden.shape[1]
        query_indices = direct_cp_metadata.collective_block_ids.to(query.device)
        query_indices = (
            query_indices[:, None] * metadata.block_size
            + torch.arange(metadata.block_size, device=query.device)[None]
        ).flatten()
        from aspec.models.ops.distributed_sparse import max_context_shard_length

        context_capacity = max_context_shard_length(
            direct_cp_metadata, context_hidden.shape[1], alignment=self.shard_alignment
        )
        group_capacity = context_capacity // self.key_compressor.ratio
        bytes_per_query = max(
            query.shape[0]
            * max(group_capacity, 1)
            * torch.float32.itemsize
            * (self.num_heads + 3),
            1,
        )
        chunk_size = max(1, workspace_bytes // bytes_per_query)
        losses = []
        for start in range(0, query.shape[1], chunk_size):
            end = min(start + chunk_size, query.shape[1])
            eligible = csa_eligible_groups(
                metadata,
                layout,
                start,
                end,
                query_indices=query_indices,
            )
            scores = torch.einsum(
                "bqhd,bcd->bqhc",
                query[:, start:end].float(),
                index_key.float(),
            ).relu_()
            scores = (scores * weights[:, start:end].unsqueeze(-1)).sum(dim=2)
            log_probability, valid = distributed_log_softmax(
                scores,
                eligible,
                group=direct_cp_metadata.group,
            )
            target, main_valid = _distributed_csa_full_target(
                query=main_query[:, start:end].detach(),
                compressed_key=main_compressed_key.detach(),
                eligible=eligible,
                scale=scale,
                group=direct_cp_metadata.group,
            )
            local_loss = F.kl_div(
                log_probability, target, reduction="none"
            ).sum(dim=-1)
            if direct_cp_metadata.cp_size > 1:
                from torch.distributed.nn.functional import all_reduce

                loss = all_reduce(local_loss, group=direct_cp_metadata.group)
            else:
                loss = local_loss
            losses.append(
                torch.where(valid & main_valid, loss, torch.zeros_like(loss))
            )
        loss = torch.cat(losses, dim=1)
        rank = (
            dist.get_rank(direct_cp_metadata.group)
            if direct_cp_metadata.cp_size > 1 else 0
        )
        local_start = rank * local_query_count
        return loss[:, local_start : local_start + local_query_count]

    def forward_sparse_bounded(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        layout: CSACompressionLayout,
        metadata: CSAForwardMetadata,
        main_query: torch.Tensor,
        main_compressed_key: torch.Tensor,
        scale: float,
        workspace_bytes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CP1 sparse selection without retaining a query-by-context score matrix."""

        draft_input = draft_hidden.detach()
        _, index_key = self.key_compressor(context_hidden.detach(), layout)
        index_key = index_key.squeeze(2)
        query = self.q_proj(draft_input).view(
            *draft_input.shape[:2], self.num_heads, self.head_dim
        )
        query = self.rotary(query, draft_position_ids)
        weights = self.weights_proj(draft_input).float() * self.scale
        bytes_per_query = max(
            query.shape[0]
            * max(layout.num_groups, 1)
            * torch.float32.itemsize
            * 2,
            1,
        )
        chunk_size = max(1, workspace_bytes // bytes_per_query)
        selections = []
        losses = []
        for start in range(0, query.shape[1], chunk_size):
            end = min(start + chunk_size, query.shape[1])
            eligible = csa_eligible_groups(metadata, layout, start, end)
            scores = torch.einsum(
                "bqhd,bcd->bqhc",
                query[:, start:end].float(),
                index_key.float(),
            ).relu_()
            scores = (scores * weights[:, start:end].unsqueeze(-1)).sum(dim=2)
            scores = scores.masked_fill(~eligible, -torch.inf)
            topk = min(self.topk, scores.shape[-1])
            if topk:
                indices = scores.topk(topk, dim=-1, sorted=False).indices
                selected_valid = torch.gather(eligible, -1, indices)
                indices = torch.where(
                    selected_valid, indices, torch.full_like(indices, -1)
                )
                selected_scores = torch.gather(scores, -1, indices.clamp_min(0))
            else:
                indices = torch.empty(
                    *scores.shape[:2], 0, dtype=torch.long, device=scores.device
                )
                selected_valid = torch.empty_like(indices, dtype=torch.bool)
                selected_scores = scores.new_empty(*scores.shape[:2], 0)
            if not getattr(self, "compute_auxiliary_loss", True):
                selections.append(indices)
                losses.append(scores.new_zeros(scores.shape[:2]))
                continue
            target, main_valid = _selected_main_attention_target(
                main_query[:, start:end].detach(),
                main_compressed_key.detach(),
                indices,
                scale,
                query_chunk_size=max(1, end - start),
            )
            log_prob, valid = _masked_log_softmax(selected_scores, selected_valid)
            loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
            selections.append(indices)
            losses.append(torch.where(valid & main_valid, loss, torch.zeros_like(loss)))
        return torch.cat(selections, dim=1), torch.cat(losses, dim=1)

    def loss_per_query(
        self,
        *,
        index_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        main_query: torch.Tensor,
        main_compressed_key: torch.Tensor,
        eligible: torch.Tensor,
        stage: str,
        scale: float,
    ) -> torch.Tensor:
        if stage == "indexer_warmup":
            target = _main_attention_target(
                main_query.detach(), main_compressed_key.detach(), eligible, scale
            )
            log_prob, valid = _masked_log_softmax(index_scores, eligible)
            loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
            return torch.where(valid, loss, torch.zeros_like(loss))
        if stage != "sparse":
            return index_scores.new_zeros(index_scores.shape[:2])

        selected_valid = topk_indices >= 0
        selected_logits = _gather_last(index_scores, topk_indices)
        target, main_valid = _selected_main_attention_target(
            main_query.detach(),
            main_compressed_key.detach(),
            topk_indices,
            scale,
        )
        log_prob, valid = _masked_log_softmax(selected_logits, selected_valid)
        loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
        valid = valid & main_valid
        return torch.where(valid, loss, torch.zeros_like(loss))


def selected_group_mask(eligible: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    visible = torch.zeros_like(eligible)
    if indices.shape[-1]:
        valid = indices >= 0
        visible.scatter_(2, indices.clamp_min(0), valid)
    return visible


def _distributed_csa_token_rows(
    *,
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    local_selected: torch.Tensor,
    direct_cp_metadata,
    query_indices: torch.Tensor,
    raw_key_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build owner-local raw-window, Draft-block and compressed CSA rows."""

    from aspec.models.ops.distributed_sparse import owner_local_rows

    device = local_selected.device
    batch, query_count, selected_count = local_selected.shape
    query_indices = query_indices.to(device=device, dtype=torch.long)
    if query_indices.numel() != query_count:
        raise ValueError("CSA distributed query ids must match selected rows")
    anchors = metadata.query_anchors.to(device).index_select(1, query_indices)
    keep = metadata.query_keep.to(device).index_select(1, query_indices)
    window_offsets = torch.arange(metadata.context_window, device=device).view(1, 1, -1)
    window_physical = anchors.unsqueeze(-1) - metadata.context_window + window_offsets
    window_local, window_owned = owner_local_rows(
        direct_cp_metadata.context_position_ids.to(device), window_physical
    )
    window_valid = window_owned & window_physical.lt(anchors.unsqueeze(-1)) & keep.unsqueeze(-1)
    if metadata.context_doc_ids is not None and metadata.context_doc_ids.shape[1]:
        local_docs = torch.gather(
            metadata.context_doc_ids.to(device),
            1,
            window_local.flatten(1),
        ).view_as(window_local)
        query_docs = metadata.query_docs.to(device).index_select(1, query_indices)
        window_valid &= query_docs.unsqueeze(-1).ge(0) & local_docs.eq(
            query_docs.unsqueeze(-1)
        )

    query_blocks = torch.div(
        query_indices, metadata.block_size, rounding_mode="floor"
    )
    local_block_ids = direct_cp_metadata.local_block_ids.to(device)
    global_to_local = torch.full(
        (direct_cp_metadata.num_blocks,), -1, dtype=torch.long, device=device
    )
    global_to_local[local_block_ids] = torch.arange(
        local_block_ids.numel(), device=device
    )
    local_blocks = global_to_local[query_blocks]
    draft_offsets = torch.arange(metadata.block_size, device=device).view(1, 1, -1)
    draft_indices = (
        direct_cp_metadata.context_length
        + local_blocks.clamp_min(0).view(1, -1, 1) * metadata.block_size
        + draft_offsets
    ).expand(batch, -1, -1)
    draft_valid = (
        local_blocks.ge(0).view(1, -1, 1)
        & keep.unsqueeze(-1)
    ).expand_as(draft_indices)

    if layout.num_groups:
        compressed_indices = raw_key_length + local_selected.clamp_min(0)
        compressed_valid = local_selected.ge(0) & torch.gather(
            layout.valid,
            1,
            local_selected.clamp_min(0).flatten(1),
        ).view_as(local_selected)
    else:
        compressed_indices = torch.full_like(local_selected, raw_key_length)
        compressed_valid = torch.zeros_like(local_selected, dtype=torch.bool)
    compressed_valid &= keep.unsqueeze(-1)
    return (
        torch.cat((window_local, draft_indices, compressed_indices), dim=-1),
        torch.cat((window_valid, draft_valid, compressed_valid), dim=-1),
        keep,
    )


def csa_distributed_sparse_attention(
    *,
    query_own: torch.Tensor,
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    compressed_key: torch.Tensor,
    compressed_value: torch.Tensor,
    local_selected: torch.Tensor,
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    direct_cp_metadata,
    scale: float,
    workspace_bytes: int,
) -> torch.Tensor:
    """Run CSA with bounded metadata and no context/compressed-key all-gather."""

    from aspec.models.ops.compact_varlen import distributed_compact_varlen_attention
    from aspec.models.ops.direct_cp import build_context_positions

    cp_size = direct_cp_metadata.cp_size
    key = torch.cat((raw_key, compressed_key), dim=1)
    value = torch.cat((raw_value, compressed_value), dim=1)
    capacity = metadata.context_window + metadata.block_size + local_selected.shape[-1]
    metadata_bytes_per_local_query = max(
        query_own.shape[0] * cp_size * capacity * 9, 1
    )
    local_chunk = max(1, workspace_bytes // metadata_bytes_per_local_query)
    draft_offsets = torch.arange(
        metadata.block_size, device=query_own.device
    ).view(1, -1)
    rank_query_rows = []
    for rank in range(cp_size):
        rank_blocks = build_context_positions(
            direct_cp_metadata.num_blocks, rank, cp_size, layout="contiguous"
        ).to(query_own.device)
        rank_query_rows.append(
            (rank_blocks[:, None] * metadata.block_size + draft_offsets).flatten()
        )
    outputs = []
    for start in range(0, query_own.shape[1], local_chunk):
        end = min(start + local_chunk, query_own.shape[1])
        query_indices = torch.cat(
            [rank_queries[start:end] for rank_queries in rank_query_rows]
        )
        selected = local_selected.index_select(1, query_indices)
        token_indices, token_valid, keep = _distributed_csa_token_rows(
            metadata=metadata,
            layout=layout,
            local_selected=selected,
            direct_cp_metadata=direct_cp_metadata,
            query_indices=query_indices,
            raw_key_length=raw_key.shape[1],
        )
        output, _ = distributed_compact_varlen_attention(
            query_own=query_own[:, start:end],
            key=key,
            value=value,
            token_indices=token_indices,
            token_valid=token_valid,
            sequence_keep=keep,
            queries_per_sequence=1,
            scale=scale,
            workspace_bytes=workspace_bytes,
            group=direct_cp_metadata.group,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1)


@lru_cache(maxsize=None)
def _build_csa_compressed_mask(
    *,
    block_size: int,
    num_blocks: int,
    query_length: int,
    key_length: int,
    packed: bool,
):
    """Compile CSA's structured compressed-prefix predicate once per shape."""

    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def read_aux(aux, batch, index: "cute.TensorSSA", limit: int):
        index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        index_fragment.store(index)
        value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        value_fragment[0] = aux[
            batch[0], cutlass.min(index_fragment[0], limit - 1)
        ]
        return value_fragment.load()

    @cute.jit
    def mask_mod(batch, head, query_index, key_index, seqlen_info, aux_tensors):
        del head, seqlen_info
        (
            anchors,
            query_blocks,
            query_keep,
            compressed_ends,
            compressed_valid,
            anchor_docs,
            compressed_docs,
        ) = aux_tensors
        zero = cutlass.Int32(0)
        query_block = read_aux(query_blocks, batch, query_index, query_length)
        anchor = read_aux(anchors, batch, query_block, num_blocks)
        keep = read_aux(query_keep, batch, query_index, query_length)
        end = read_aux(compressed_ends, batch, key_index, key_length)
        valid = read_aux(compressed_valid, batch, key_index, key_length)
        visible = (keep > zero) & (valid > zero) & (end <= anchor)
        if cutlass.const_expr(packed):
            query_doc = read_aux(anchor_docs, batch, query_block, num_blocks)
            key_doc = read_aux(compressed_docs, batch, key_index, key_length)
            visible = visible & (query_doc >= zero) & (query_doc == key_doc)
        return visible

    mask_mod.__name__ = (
        f"dflash_csa_compressed_B{block_size}_Q{query_length}_K{key_length}_"
        f"P{int(packed)}"
    )
    return mask_mod


def _csa_compressed_prefix_attention(
    *,
    query_own: torch.Tensor,
    compressed_key: torch.Tensor,
    compressed_value: torch.Tensor,
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    direct_cp_metadata,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to owner-local CSA prefixes with O(Q+K) mask metadata."""

    from flash_attn.cute.interface import flash_attn_func

    from aspec.models.ops.direct_cp import all_gather_query_rows
    from aspec.models.ops.fa4_sparse import build_csa_prefix_sparse_pair

    query_full = (
        query_own
        if direct_cp_metadata.cp_size == 1
        else all_gather_query_rows(query_own, group=direct_cp_metadata.group)
    )
    batch, query_length = query_full.shape[:2]
    key_length = compressed_key.shape[1]
    if key_length == 0:
        # No local keys still means one zero partial for every global query,
        # followed by the same forward/backward exchanges as nonempty owners.
        zero = (
            query_full.reshape(-1)[:0].sum()
            + compressed_key.sum() + compressed_value.sum()
        )
        partial_output = query_full.new_zeros(
            *query_full.shape[:-1], compressed_value.shape[-1]
        ) + zero
        partial_lse = torch.full(
            (batch, query_own.shape[2], query_length), -torch.inf,
            dtype=torch.float32, device=query_own.device,
        ) + zero.float()
        return _merge_csa_prefix_partials(
            partial_output, partial_lse, direct_cp_metadata
        )
    if key_length == 1:
        # FA4 cannot infer a leading dimension for a [1, 1] auxiliary tensor:
        # both dimensions have stride one.  Add one explicitly-invalid key
        # column instead of constraining CP degree or padding the raw context.
        compressed_key = torch.cat(
            (compressed_key, torch.zeros_like(compressed_key[:, :1])), dim=1
        )
        compressed_value = torch.cat(
            (compressed_value, torch.zeros_like(compressed_value[:, :1])), dim=1
        )
        key_length = 2
    query_blocks = direct_cp_metadata.collective_block_ids.to(
        query_own.device
    ).repeat_interleave(metadata.block_size)
    query_blocks = query_blocks.unsqueeze(0).expand(batch, -1).to(torch.int32)
    anchors = metadata.anchor_positions.to(
        device=query_own.device, dtype=torch.int32
    )
    keep = metadata.block_keep_mask.to(query_own.device).bool()
    if direct_cp_metadata.sequence_length is not None:
        offsets = torch.arange(
            metadata.block_size, device=query_own.device
        ).view(1, 1, -1)
        keep_tokens = (
            keep.unsqueeze(-1)
            & (
                metadata.anchor_positions.to(query_own.device).unsqueeze(-1)
                + offsets
                < direct_cp_metadata.sequence_length
            )
        ).flatten(1)
    else:
        keep_tokens = keep.repeat_interleave(metadata.block_size, dim=1)
    keep_tokens = keep_tokens.index_select(
        1,
        direct_cp_metadata.collective_block_ids.to(query_own.device)
        .unsqueeze(-1)
        .mul(metadata.block_size)
        .add(torch.arange(metadata.block_size, device=query_own.device))
        .flatten(),
    ).to(torch.int32)
    compressed_ends = layout.physical_ends.to(
        device=query_own.device, dtype=torch.int32
    ).contiguous()
    compressed_valid = layout.valid.to(
        device=query_own.device, dtype=torch.int32
    ).contiguous()
    if compressed_ends.shape[1] == 1:
        compressed_ends = torch.cat(
            (compressed_ends, torch.zeros_like(compressed_ends)), dim=1
        )
        compressed_valid = torch.cat(
            (compressed_valid, torch.zeros_like(compressed_valid)), dim=1
        )
    if metadata.anchor_doc_ids is None:
        anchor_docs = torch.zeros_like(anchors)
        compressed_docs = torch.zeros_like(compressed_ends)
    else:
        anchor_docs = metadata.anchor_doc_ids.to(
            device=query_own.device, dtype=torch.int32
        ).contiguous()
        compressed_docs = layout.doc_ids.to(
            device=query_own.device, dtype=torch.int32
        ).contiguous()
        if compressed_docs.shape[1] == 1:
            compressed_docs = torch.cat(
                (compressed_docs, torch.full_like(compressed_docs, -1)), dim=1
            )
    aux_tensors = [
        anchors.contiguous(),
        query_blocks.contiguous(),
        keep_tokens.contiguous(),
        compressed_ends,
        compressed_valid,
        anchor_docs,
        compressed_docs,
    ]
    mask_mod = _build_csa_compressed_mask(
        block_size=metadata.block_size,
        num_blocks=metadata.anchor_positions.shape[1],
        query_length=query_length,
        key_length=key_length,
        packed=metadata.anchor_doc_ids is not None,
    )
    sparse_key = (
        "csa_compressed_prefix",
        query_own.device,
        metadata.block_size,
        query_length,
        key_length,
        query_own.shape[-1],
        compressed_value.shape[-1],
        metadata.anchor_doc_ids is not None,
    )
    sparse_pair = direct_cp_metadata._fa4_cache.get(sparse_key)
    if sparse_pair is None:
        sparse_pair = build_csa_prefix_sparse_pair(
            anchor_positions=anchors,
            query_blocks=query_blocks,
            compressed_ends=compressed_ends,
            query_token_keep=keep_tokens,
            compressed_valid=compressed_valid,
            query_head_dim=query_own.shape[-1],
            value_head_dim=compressed_value.shape[-1],
            anchor_doc_ids=(
                anchor_docs if metadata.anchor_doc_ids is not None else None
            ),
            compressed_doc_ids=(
                compressed_docs if metadata.anchor_doc_ids is not None else None
            ),
        )
        direct_cp_metadata._fa4_cache[sparse_key] = sparse_pair
    partial_output, partial_lse, *_ = flash_attn_func(
        query_full,
        compressed_key,
        compressed_value,
        softmax_scale=scale,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=sparse_pair.forward,
        block_sparse_tensors_bwd=sparse_pair.backward,
        return_lse=True,
    )
    if partial_lse.shape != (batch, query_own.shape[2], query_length):
        partial_lse = partial_lse.reshape(
            batch, query_own.shape[2], query_length
        )
    return _merge_csa_prefix_partials(
        partial_output, partial_lse, direct_cp_metadata
    )


def _merge_csa_prefix_partials(partial_output, partial_lse, direct_cp_metadata):
    from aspec.models.ops.direct_cp import (
        all_to_all_query_partials,
        merge_attention_partials,
    )

    if direct_cp_metadata.cp_size == 1:
        return partial_output, partial_lse
    output_parts = all_to_all_query_partials(
        partial_output.permute(0, 2, 1, 3),
        group=direct_cp_metadata.group,
    )
    lse_parts = all_to_all_query_partials(
        partial_lse,
        group=direct_cp_metadata.group,
    )
    output, lse = merge_attention_partials(output_parts, lse_parts)
    return output.permute(0, 2, 1, 3).to(partial_output.dtype), lse


def csa_direct_dense_attention(
    *,
    query_own: torch.Tensor,
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    compressed_key: torch.Tensor,
    compressed_value: torch.Tensor,
    metadata: CSAForwardMetadata,
    layout: CSACompressionLayout,
    direct_cp_metadata,
    scale: float,
) -> torch.Tensor:
    """Merge the structured raw-window and compressed-prefix CSA branches."""

    from aspec.models.ops.direct_cp import (
        direct_cp_attention_from_kv,
        merge_attention_partials,
    )

    raw_output, raw_lse = direct_cp_attention_from_kv(
        query_own.permute(0, 2, 1, 3),
        raw_key.permute(0, 2, 1, 3),
        raw_value.permute(0, 2, 1, 3),
        direct_cp_metadata,
        scale,
    )
    compressed_output, compressed_lse = _csa_compressed_prefix_attention(
        query_own=query_own,
        compressed_key=compressed_key,
        compressed_value=compressed_value,
        metadata=metadata,
        layout=layout,
        direct_cp_metadata=direct_cp_metadata,
        scale=scale,
    )
    output, _ = merge_attention_partials(
        torch.stack(
            (raw_output, compressed_output.permute(0, 2, 1, 3)), dim=0
        ),
        torch.stack((raw_lse, compressed_lse), dim=0),
    )
    local_blocks = direct_cp_metadata.local_block_ids.to(query_own.device)
    keep = metadata.block_keep_mask.to(query_own.device).index_select(
        1, local_blocks
    ).repeat_interleave(metadata.block_size, dim=1)
    return torch.where(
        keep[:, :, None, None],
        output.permute(0, 2, 1, 3).to(query_own.dtype),
        torch.zeros_like(query_own),
    )


def build_csa_block_mask(
    *,
    metadata: CSAForwardMetadata,
    compressed_visible: torch.Tensor,
    context_length: int,
    draft_length: int,
    device: torch.device,
):
    """Build the exact DFlash local-window plus arbitrary compressed-entry mask."""

    from aspec.models.ops.flex_attention import compile_friendly_create_block_mask

    raw_key_length = context_length + draft_length
    block_size = metadata.block_size
    anchors = metadata.anchor_positions.to(device)
    keep = metadata.block_keep_mask.to(device)
    context_docs = (
        metadata.context_doc_ids.to(device) if metadata.context_doc_ids is not None else None
    )
    anchor_docs = (
        metadata.anchor_doc_ids.to(device) if metadata.anchor_doc_ids is not None else None
    )

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        query_block = q_idx // block_size
        anchor = anchors[b, query_block]
        is_context = kv_idx < context_length
        visible_context = is_context & (kv_idx < anchor) & (
            kv_idx >= anchor - metadata.context_window
        )
        if context_docs is not None:
            safe_context = torch.minimum(kv_idx, context_length - 1)
            visible_context = visible_context & (
                anchor_docs[b, query_block] >= 0
            ) & (context_docs[b, safe_context] == anchor_docs[b, query_block])
        is_draft = (kv_idx >= context_length) & (kv_idx < raw_key_length)
        draft_block = (kv_idx - context_length) // block_size
        visible_draft = is_draft & (draft_block == query_block)
        if compressed_visible.shape[-1]:
            compressed_index = torch.clamp(
                kv_idx - raw_key_length,
                min=0,
                max=compressed_visible.shape[-1] - 1,
            )
            visible_compressed = (kv_idx >= raw_key_length) & compressed_visible[
                b, q_idx, compressed_index
            ]
        else:
            visible_compressed = kv_idx < 0
        return (visible_context | visible_draft | visible_compressed) & keep[b, query_block]

    mask_mod.__name__ = (
        f"dflash_csa_A{anchors.shape[1]}_B{block_size}_C{context_length}_"
        f"G{compressed_visible.shape[-1]}"
    )
    return compile_friendly_create_block_mask(
        mask_mod=mask_mod,
        B=anchors.shape[0],
        H=None,
        Q_LEN=draft_length,
        KV_LEN=raw_key_length + compressed_visible.shape[-1],
        device=device,
    )


def csa_compact_attention(
    *,
    query: torch.Tensor,
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    compressed_key: torch.Tensor,
    compressed_value: torch.Tensor,
    compressed_visible: torch.Tensor,
    metadata: CSAForwardMetadata,
    scale: float,
    workspace_bytes: int,
) -> torch.Tensor:
    """Execute CSA from exact compact keys without rebuilding a padded batch."""

    from aspec.models.ops.compact_varlen import compact_varlen_attention

    batch, query_length = query.shape[:2]
    context_length = raw_key.shape[1] - query_length
    if context_length < 0 or raw_value.shape[1] != raw_key.shape[1]:
        raise ValueError("CSA raw key/value sequence is inconsistent")
    if compressed_key.shape[:3] != compressed_value.shape[:3]:
        raise ValueError("CSA compressed key/value leading shapes must match")
    if compressed_visible.shape != (
        batch,
        query_length,
        compressed_key.shape[1],
    ):
        raise ValueError("CSA compressed visibility does not match queries and groups")

    device = query.device
    query_ids = torch.arange(query_length, device=device)
    query_blocks = torch.div(query_ids, metadata.block_size, rounding_mode="floor")
    anchors = metadata.query_anchors.to(device)
    keep = metadata.query_keep.to(device)
    window_offsets = torch.arange(metadata.context_window, device=device)
    window_start = anchors - metadata.context_window
    context_indices = window_start.unsqueeze(-1) + window_offsets
    context_valid = (
        (context_indices >= 0)
        & (context_indices < anchors.unsqueeze(-1))
        & keep.unsqueeze(-1)
    )
    safe_context = context_indices.clamp(0, max(context_length - 1, 0))
    if metadata.context_doc_ids is not None:
        context_docs = metadata.context_doc_ids.to(device)
        query_docs = metadata.query_docs.to(device)
        gathered_docs = torch.gather(context_docs, 1, safe_context.flatten(1)).view_as(
            safe_context
        )
        context_valid &= (
            (query_docs.unsqueeze(-1) >= 0)
            & (gathered_docs == query_docs.unsqueeze(-1))
        )

    draft_offsets = torch.arange(metadata.block_size, device=device)
    draft_indices = (
        context_length
        + query_blocks.view(1, -1, 1) * metadata.block_size
        + draft_offsets.view(1, 1, -1)
    ).expand(batch, -1, -1)
    draft_valid = (
        (draft_indices < raw_key.shape[1]) & keep.unsqueeze(-1)
    )
    compressed_indices = (
        raw_key.shape[1]
        + torch.arange(compressed_key.shape[1], device=device).view(1, 1, -1)
    ).expand(batch, query_length, -1)
    token_indices = torch.cat(
        (safe_context, draft_indices, compressed_indices), dim=-1
    )
    token_valid = torch.cat(
        (context_valid, draft_valid, compressed_visible.to(device)), dim=-1
    )
    key = torch.cat((raw_key, compressed_key), dim=1)
    value = torch.cat((raw_value, compressed_value), dim=1)
    return compact_varlen_attention(
        query=query,
        key=key,
        value=value,
        token_indices=token_indices,
        token_valid=token_valid,
        sequence_keep=keep,
        queries_per_sequence=1,
        scale=scale,
        workspace_bytes=workspace_bytes,
    )


class CSALayer(nn.Module):
    """Per-layer CSA compressor and indexer; no CSA2 cross-layer reuse."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        ratio = config.compress_ratios[layer_idx]
        self.layer_idx = layer_idx
        self.ratio = ratio
        self.stage = config.csa_training_stage
        self.indexer_loss_coef = config.csa_indexer_loss_coef
        self.workspace_bytes = config.csa_workspace_bytes
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.compressor = CSAOverlappingCompressor(
            hidden_size=config.hidden_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            ratio=ratio,
            rope_theta=config.compress_rope_theta,
            norm_eps=config.rms_norm_eps,
        )
        self.indexer = CSALightningIndexer(config, layer_idx)

    def keep_projection_fp32(self) -> None:
        self.compressor.keep_projection_fp32()
        self.indexer.keep_projection_fp32()

    def configure_trainable_parameters(self) -> None:
        if self.stage != "indexer_warmup":
            return
        # The parent model freezes everything first and then calls this method.
        for parameter in self.indexer.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        metadata: CSAForwardMetadata,
        main_query: torch.Tensor,
        scale: float,
        layout: CSACompressionLayout | None = None,
        eligible: torch.Tensor | None = None,
        direct_cp_metadata=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        distributed = (
            direct_cp_metadata is not None and direct_cp_metadata.cp_size > 1
        )
        bounded_direct_sparse = (
            direct_cp_metadata is not None and self.stage == "sparse"
        )
        if direct_cp_metadata is None and (layout is None) != (eligible is None):
            raise ValueError("CSA prepared layout and eligibility must be provided together")
        if layout is None:
            layout = build_csa_compression_layout(metadata, context_position_ids, self.ratio)
        if eligible is None and direct_cp_metadata is None:
            eligible = csa_eligible_groups(metadata, layout)
        remote_predecessor = _csa_remote_predecessor_halo(
            context_hidden=context_hidden,
            metadata=metadata,
            layout=layout,
            ratio=self.ratio,
            direct_cp_metadata=direct_cp_metadata,
        )
        # The upstream MLA latent is rotated as a key/value and inverse-rotated
        # after attention.  DFlash uses standard GQA with distinct K/V, so its
        # faithful extension is a rotated compressed key and an unrotated value.
        compressed_value, compressed_key = self.compressor(
            context_hidden,
            layout,
            remote_predecessor=remote_predecessor,
        )
        per_query_loss = main_query.new_zeros(main_query.shape[:2], dtype=torch.float32)
        topk = None
        if not getattr(self, "compute_auxiliary_loss", True) and self.stage != "sparse":
            pass  # Dense/warmup proposals use full compressed attention, no indexer KL.
        elif distributed:
            if self.stage == "sparse":
                topk, per_query_loss = self.indexer.forward_distributed_sparse(
                    draft_hidden=draft_hidden,
                    context_hidden=context_hidden,
                    draft_position_ids=draft_position_ids,
                    layout=layout,
                    metadata=metadata,
                    main_query=main_query,
                    main_compressed_key=compressed_key,
                    scale=scale,
                    workspace_bytes=self.workspace_bytes,
                    direct_cp_metadata=direct_cp_metadata,
                    remote_predecessor=remote_predecessor,
                )
            elif self.stage == "indexer_warmup":
                per_query_loss = self.indexer.forward_distributed_warmup(
                    draft_hidden=draft_hidden,
                    context_hidden=context_hidden,
                    draft_position_ids=draft_position_ids,
                    layout=layout,
                    metadata=metadata,
                    main_query=main_query,
                    main_compressed_key=compressed_key,
                    scale=scale,
                    workspace_bytes=self.workspace_bytes,
                    direct_cp_metadata=direct_cp_metadata,
                    remote_predecessor=remote_predecessor,
                )
        elif bounded_direct_sparse:
            topk, per_query_loss = self.indexer.forward_sparse_bounded(
                draft_hidden=draft_hidden,
                context_hidden=context_hidden,
                draft_position_ids=draft_position_ids,
                layout=layout,
                metadata=metadata,
                main_query=main_query,
                main_compressed_key=compressed_key,
                scale=scale,
                workspace_bytes=self.workspace_bytes,
            )
        elif direct_cp_metadata is not None and self.stage == "indexer_warmup":
            per_query_loss = self.indexer.forward_distributed_warmup(
                draft_hidden=draft_hidden,
                context_hidden=context_hidden,
                draft_position_ids=draft_position_ids,
                layout=layout,
                metadata=metadata,
                main_query=main_query,
                main_compressed_key=compressed_key,
                scale=scale,
                workspace_bytes=self.workspace_bytes,
                direct_cp_metadata=direct_cp_metadata,
                remote_predecessor=remote_predecessor,
            )
        elif self.stage in {"indexer_warmup", "sparse"}:
            scores, topk = self.indexer(
                draft_hidden, context_hidden, draft_position_ids, layout, eligible
            )
            per_query_loss = self.indexer.loss_per_query(
                index_scores=scores,
                topk_indices=topk,
                main_query=main_query,
                main_compressed_key=compressed_key,
                eligible=eligible,
                stage=self.stage,
                scale=scale,
            )
        visible = (
            topk
            if direct_cp_metadata is not None else (
                selected_group_mask(eligible, topk)
                if self.stage == "sparse"
                else eligible
            )
        )
        observation = {
            "layer": self.layer_idx,
            "type": "compressed_sparse_attention",
            "variant": "csa",
            "training_stage": self.stage,
            "compress_ratio": self.ratio,
            "compressed_group_capacity": layout.num_groups,
            "index_topk": self.indexer.topk,
            "indexer_input_detached": True,
            "cross_layer_reuse": False,
        }
        return compressed_key, compressed_value, visible, per_query_loss, observation


__all__ = [
    "CSAForwardMetadata",
    "CSACompressionLayout",
    "CSAOverlappingCompressor",
    "CSALightningIndexer",
    "CSALayer",
    "build_csa_block_mask",
    "build_csa_compression_layout",
    "csa_compact_attention",
    "csa_eligible_groups",
    "selected_group_mask",
]
