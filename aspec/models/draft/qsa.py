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

"""Qwen Sparse Attention adapted to the DSpark/DFlash Draft path.

QSA compresses keys only for its lightweight block indexer.  The core
attention still consumes the selected original context tokens, together with
the incomplete micro-block tail and the complete current Draft block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QSAForwardMetadata:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    block_size: int
    context_doc_ids: torch.Tensor | None = None
    anchor_doc_ids: torch.Tensor | None = None
    context_physical_position_ids: torch.Tensor | None = None
    anchor_rope_position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.anchor_positions.ndim != 2:
            raise ValueError("QSA anchor_positions must be [batch, anchors]")
        if self.block_keep_mask.shape != self.anchor_positions.shape:
            raise ValueError("QSA block_keep_mask must match anchor_positions")
        if self.block_size <= 0:
            raise ValueError("QSA block_size must be positive")
        if (self.context_doc_ids is None) != (self.anchor_doc_ids is None):
            raise ValueError("QSA context_doc_ids and anchor_doc_ids must be provided together")
        if self.context_doc_ids is not None:
            if self.context_doc_ids.ndim != 2:
                raise ValueError("QSA context_doc_ids must be [batch, context]")
            if self.context_doc_ids.shape[0] != self.anchor_positions.shape[0]:
                raise ValueError("QSA context and anchor batches must match")
            if self.anchor_doc_ids.shape != self.anchor_positions.shape:
                raise ValueError("QSA anchor_doc_ids must match anchor_positions")
        if self.context_physical_position_ids is not None:
            if self.context_physical_position_ids.ndim != 2:
                raise ValueError("QSA physical context positions must be [batch, context]")
            if (
                self.context_doc_ids is not None
                and self.context_physical_position_ids.shape
                != self.context_doc_ids.shape
            ):
                raise ValueError("QSA physical positions must match the local context")
        if (
            self.anchor_rope_position_ids is not None
            and self.anchor_rope_position_ids.shape != self.anchor_positions.shape
        ):
            raise ValueError("QSA anchor RoPE positions must match anchor positions")

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
class QSACompressionLayout:
    token_indices: torch.Tensor
    valid: torch.Tensor
    physical_starts: torch.Tensor
    physical_ends: torch.Tensor
    rope_positions: torch.Tensor
    doc_ids: torch.Tensor
    context_group_ids: torch.Tensor
    context_run_offsets: torch.Tensor

    @property
    def num_groups(self) -> int:
        return self.token_indices.shape[1]


def build_qsa_compression_layout(
    metadata: QSAForwardMetadata,
    context_rope_position_ids: torch.Tensor,
    ratio: int,
) -> QSACompressionLayout:
    """Build complete, non-overlapping micro-blocks inside packed documents."""

    if ratio < 2:
        raise ValueError("QSA compression ratio must be at least two")
    if context_rope_position_ids.ndim != 2:
        raise ValueError("QSA context RoPE positions must be [batch, context]")
    batch, context_length = context_rope_position_ids.shape
    if batch != metadata.anchor_positions.shape[0]:
        raise ValueError("QSA context and anchor batches must match")
    device = context_rope_position_ids.device
    if metadata.context_doc_ids is None:
        doc_ids = torch.zeros(batch, context_length, dtype=torch.long, device=device)
    else:
        if metadata.context_doc_ids.shape != (batch, context_length):
            raise ValueError("QSA context_doc_ids must match the context sequence")
        doc_ids = metadata.context_doc_ids.to(device=device, dtype=torch.long)

    positions = torch.arange(context_length, device=device, dtype=torch.long).view(1, -1)
    distributed_positions = metadata.context_physical_position_ids is not None
    physical_positions = metadata.context_physical_position_ids
    if not distributed_positions:
        physical_positions = positions.expand(batch, -1)
    elif physical_positions.shape != (batch, context_length):
        raise ValueError("QSA physical positions must match the local context")
    else:
        physical_positions = physical_positions.to(device=device, dtype=torch.long)
    boundary = torch.ones(batch, context_length, dtype=torch.bool, device=device)
    if context_length > 1:
        boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    run_starts = torch.cummax(
        torch.where(boundary, positions, torch.zeros_like(positions)), dim=1
    ).values
    run_offsets = positions - run_starts
    max_groups = context_length // ratio
    if max_groups == 0:
        empty = torch.empty(batch, 0, dtype=torch.long, device=device)
        return QSACompressionLayout(
            token_indices=torch.empty(batch, 0, ratio, dtype=torch.long, device=device),
            valid=empty.bool(),
            physical_starts=empty,
            physical_ends=empty,
            rope_positions=empty,
            doc_ids=empty,
            context_group_ids=torch.full(
                (batch, context_length), -1, dtype=torch.long, device=device
            ),
            context_run_offsets=run_offsets.expand(batch, -1),
        )

    candidate = (doc_ids >= 0) & (
        context_rope_position_ids.remainder(ratio).eq(0)
        if distributed_positions else run_offsets.remainder(ratio).eq(0)
    )
    ends = positions + ratio - 1
    safe_ends = ends.clamp(max=context_length - 1)
    end_docs = torch.gather(doc_ids, 1, safe_ends.expand(batch, -1))
    end_rope = torch.gather(
        context_rope_position_ids, 1, safe_ends.expand(batch, -1)
    )
    candidate &= (ends < context_length) & (end_docs == doc_ids)
    if distributed_positions:
        candidate &= end_rope == context_rope_position_ids + ratio - 1
    compact_ids = candidate.long().cumsum(dim=1) - 1
    starts = torch.full((batch, max_groups), -1, dtype=torch.long, device=device)
    starts.scatter_reduce_(
        1,
        compact_ids.clamp_min(0),
        torch.where(candidate, positions.expand(batch, -1), -1),
        reduce="amax",
        include_self=True,
    )
    valid = starts >= 0
    safe_starts = starts.clamp_min(0)
    offsets = torch.arange(ratio, device=device).view(1, 1, -1)
    token_indices = safe_starts.unsqueeze(-1) + offsets
    group_docs = torch.gather(doc_ids, 1, safe_starts)
    rope_positions = torch.gather(context_rope_position_ids, 1, safe_starts)
    context_group_ids = torch.full(
        (batch, context_length), -1, dtype=torch.long, device=device
    )
    group_ids = torch.arange(max_groups, device=device).view(1, -1, 1)
    context_group_ids.scatter_reduce_(
        1,
        token_indices.reshape(batch, -1),
        torch.where(
            valid.unsqueeze(-1), group_ids, torch.full_like(group_ids, -1)
        ).expand(-1, -1, ratio).reshape(batch, -1),
        reduce="amax",
        include_self=True,
    )
    physical_starts = torch.gather(physical_positions, 1, safe_starts)
    physical_ends = torch.gather(
        physical_positions,
        1,
        (safe_starts + ratio - 1).clamp_max(context_length - 1),
    ) + 1
    return QSACompressionLayout(
        token_indices=token_indices,
        valid=valid,
        physical_starts=torch.where(valid, physical_starts, torch.zeros_like(starts)),
        physical_ends=torch.where(valid, physical_ends, torch.zeros_like(starts)),
        rope_positions=torch.where(valid, rope_positions, torch.zeros_like(rope_positions)),
        doc_ids=torch.where(valid, group_docs, torch.full_like(group_docs, -1)),
        context_group_ids=context_group_ids,
        context_run_offsets=run_offsets.expand(batch, -1),
    )


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        values = values.float()
        values = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return values.to(dtype) * self.weight


class _PartialRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, base: float) -> None:
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("QSA rotary_dim must be a positive even integer")
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base = base
        self.frequency_dim = rotary_dim

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

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rotary = values[..., : self.rotary_dim]
        passthrough = values[..., self.rotary_dim :]
        # Elementwise FP32 arithmetic is not lowered by autocast to BF16/FP16.
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.float()
        cos = torch.cat((frequencies, frequencies), dim=-1).cos()
        sin = torch.cat((frequencies, frequencies), dim=-1).sin()
        while cos.ndim < rotary.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        cos, sin = cos.to(rotary.dtype), sin.to(rotary.dtype)
        half = self.rotary_dim // 2
        rotated = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
        rotary = rotary * cos + rotated * sin
        return torch.cat((rotary, passthrough), dim=-1)


def qsa_eligible_groups(
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    start: int = 0,
    end: int | None = None,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return complete groups inside the Draft query's legal context."""

    end = (
        metadata.query_anchors.shape[1] if query_indices is None else query_indices.numel()
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
    visible = layout.valid[:, None] & (
        layout.physical_ends[:, None] <= anchors[:, :, None]
    )
    if query_docs is not None:
        query_docs = query_docs.to(layout.doc_ids.device)
        visible &= (
            (query_docs[:, :, None] >= 0)
            & (layout.doc_ids[:, None] == query_docs[:, :, None])
        )
    return visible & keep.to(visible.device)[:, :, None]


def _safe_log_softmax(
    logits: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.any(dim=-1)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    masked = torch.where(valid.unsqueeze(-1), masked, torch.zeros_like(masked))
    return F.log_softmax(masked, dim=-1), valid


def _gather_groups(
    values: torch.Tensor, layout: QSACompressionLayout
) -> torch.Tensor:
    """Gather physical context rows without expanding the source in backward."""

    batch = values.shape[0]
    flat = layout.token_indices.reshape(batch, -1)
    gathered = torch.gather(
        values,
        1,
        flat.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
    )
    return gathered.view(
        batch, layout.num_groups, layout.token_indices.shape[-1], values.shape[-1]
    )


def _query_chunk_size(
    *, batch: int, keys: int, workspace_bytes: int, multiplier: int = 1
) -> int:
    bytes_per_query = max(batch * keys * torch.float32.itemsize * multiplier, 1)
    return max(1, workspace_bytes // bytes_per_query)


def _full_teacher_target(
    *,
    main_query: torch.Tensor,
    main_key: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    start: int,
    end: int,
    scale: float,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Qwen's head-summed token teacher, max-pooled to micro-blocks."""

    batch, _, query_heads, _ = main_query.shape
    context_length = layout.context_group_ids.shape[1]
    total_length = main_key.shape[1]
    draft_length = total_length - context_length
    num_kv_heads = main_key.shape[2]
    if query_heads % num_kv_heads:
        raise ValueError("QSA teacher query heads must be divisible by KV heads")
    repetitions = query_heads // num_kv_heads
    if query_indices is None:
        query_ids = torch.arange(start, end, device=main_query.device)
    else:
        query_ids = query_indices[start:end].to(main_query.device)
    anchors = metadata.query_anchors.index_select(1, query_ids).to(main_query.device)
    keep = metadata.query_keep.index_select(1, query_ids).to(main_query.device)
    owners = torch.div(query_ids, metadata.block_size, rounding_mode="floor")
    context_ids = torch.arange(context_length, device=main_query.device)
    visible_context = context_ids.view(1, 1, -1) < anchors.unsqueeze(-1)
    query_docs = metadata.query_docs
    if query_docs is not None:
        docs = metadata.context_doc_ids.to(main_query.device)
        qdocs = query_docs.index_select(1, query_ids).to(main_query.device)
        visible_context &= (
            (qdocs.unsqueeze(-1) >= 0) & (docs[:, None] == qdocs.unsqueeze(-1))
        )
    draft_ids = torch.arange(draft_length, device=main_query.device)
    visible_draft = (
        torch.div(draft_ids, metadata.block_size, rounding_mode="floor")
        .view(1, 1, -1)
        .eq(owners.view(1, -1, 1))
    )
    visible = torch.cat(
        (visible_context.expand(batch, -1, -1), visible_draft.expand(batch, -1, -1)),
        dim=-1,
    ) & keep.unsqueeze(-1)
    valid = visible.any(dim=-1)
    probabilities = main_query.new_zeros(
        batch, end - start, total_length, dtype=torch.float32
    )
    with torch.no_grad():
        for head in range(query_heads):
            scores = torch.einsum(
                "bqd,bkd->bqk",
                main_query[:, start:end, head].float(),
                main_key[:, :, head // repetitions].float(),
            ) * scale
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
            scores = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
            probabilities.add_(scores.softmax(dim=-1) * visible)
        if layout.num_groups == 0:
            return probabilities.new_zeros(batch, end - start, 0)
        indices = layout.token_indices[:, None].expand(-1, end - start, -1, -1)
        grouped = torch.gather(
            probabilities[:, :, :context_length], 2, indices.flatten(2)
        ).view(batch, end - start, layout.num_groups, -1)
        target = grouped.max(dim=-1).values
        eligible = qsa_eligible_groups(
            metadata, layout, start, end, query_indices=query_indices
        )
        target = target * eligible
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _distributed_full_teacher_target(
    *,
    main_query: torch.Tensor,
    main_key: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    query_indices: torch.Tensor,
    direct_cp_metadata,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owner-local QSA full-teacher groups without context all-gather."""

    batch, query_count, query_heads, head_dim = main_query.shape
    kv_heads = main_key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("QSA teacher query heads must be divisible by KV heads")
    repetitions = query_heads // kv_heads
    query_indices = query_indices.to(main_query.device, dtype=torch.long)
    block_indices = torch.div(
        query_indices, metadata.block_size, rounding_mode="floor"
    )
    anchors = metadata.anchor_positions.to(main_query.device).index_select(
        1, block_indices
    )
    keep = metadata.block_keep_mask.to(main_query.device).index_select(
        1, block_indices
    )
    context_positions = direct_cp_metadata.context_position_ids.to(
        main_query.device
    )
    visible_context = (
        context_positions[:, None, :] < anchors.unsqueeze(-1)
    ) & keep.unsqueeze(-1)
    if metadata.anchor_doc_ids is not None:
        anchor_docs = metadata.anchor_doc_ids.to(main_query.device).index_select(
            1, block_indices
        )
        context_docs = metadata.context_doc_ids.to(main_query.device)
        visible_context &= (
            anchor_docs.unsqueeze(-1).ge(0)
            & context_docs[:, None, :].eq(anchor_docs.unsqueeze(-1))
        )
    local_blocks = direct_cp_metadata.local_block_ids.to(main_query.device)
    draft_blocks = local_blocks.repeat_interleave(metadata.block_size)
    visible_draft = draft_blocks.view(1, 1, -1).eq(
        block_indices.view(1, -1, 1)
    ).expand(batch, -1, -1)
    visible_draft &= keep.unsqueeze(-1)
    visible = torch.cat((visible_context, visible_draft), dim=-1)
    grouped_query = main_query.view(
        batch, query_count, kv_heads, repetitions, head_dim
    )
    token_probability = main_query.new_zeros(
        batch, query_count, main_key.shape[1], dtype=torch.float32
    )
    world_size = direct_cp_metadata.cp_size
    # Keep all KV-head score rows in one bounded query chunk so the two
    # softmax statistics can be reduced once per chunk, rather than once per
    # KV head. At Qwen's eight KV heads the old loop issued sixteen NCCL
    # collectives before the final target reduction. The caller sizes query
    # chunks for this FP32 [KV, repetition, K] workspace.
    scores = token_probability.new_empty(
        batch,
        query_count,
        kv_heads,
        repetitions,
        main_key.shape[1],
    )
    for kv_head in range(kv_heads):
        head_scores = torch.einsum(
            "bqrd,bkd->bqrk",
            grouped_query[:, :, kv_head].float(),
            main_key[:, :, kv_head].float(),
        ) * scale
        scores[:, :, kv_head].copy_(
            head_scores.masked_fill(~visible.unsqueeze(2), -torch.inf)
        )
    if scores.shape[-1]:
        global_max = scores.amax(dim=-1)
    else:
        global_max = scores.new_full(scores.shape[:-1], -torch.inf)
    if world_size > 1:
        dist.all_reduce(
            global_max,
            op=dist.ReduceOp.MAX,
            group=direct_cp_metadata.group,
        )
    valid = torch.isfinite(global_max)
    safe_max = torch.where(valid, global_max, torch.zeros_like(global_max))
    scores.sub_(safe_max.unsqueeze(-1)).exp_()
    scores.masked_fill_(~visible[:, :, None, None], 0.0)
    denominator = scores.sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(denominator, group=direct_cp_metadata.group)
    scores.div_(denominator.clamp_min(1e-20).unsqueeze(-1))
    token_probability.add_(scores.sum(dim=(2, 3)))
    if layout.num_groups:
        grouped = torch.gather(
            token_probability[:, :, : direct_cp_metadata.context_length],
            2,
            layout.token_indices[:, None]
            .expand(-1, query_count, -1, -1)
            .flatten(2),
        ).view(batch, query_count, layout.num_groups, -1)
        # The teacher must use the same complete, visible groups as the
        # indexer. A group containing the anchor has visible tokens but is
        # still ineligible; normalize only after excluding it on every owner.
        eligible = qsa_eligible_groups(
            metadata, layout, query_indices=query_indices
        )
        target = grouped.amax(dim=-1) * eligible
    else:
        target = token_probability.new_zeros(batch, query_count, 0)
    row_sum = target.sum(dim=-1)
    if world_size > 1:
        dist.all_reduce(row_sum, group=direct_cp_metadata.group)
    row_valid = row_sum > 0
    return target / row_sum.clamp_min(1e-12).unsqueeze(-1), row_valid


def _selected_raw_tokens(
    *,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    selected: torch.Tensor,
    start: int,
    end: int,
    draft_length: int,
    query_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Expand selected blocks and append legal tail/current-Draft tokens."""

    batch, query_count, block_topk = selected.shape
    ratio = layout.token_indices.shape[-1]
    context_length = layout.context_group_ids.shape[1]
    safe = selected.clamp_min(0)
    batch_ids = torch.arange(batch, device=selected.device).view(-1, 1, 1)
    selected_tokens = layout.token_indices[batch_ids, safe]
    selected_valid = selected.ge(0).unsqueeze(-1).expand_as(selected_tokens)
    selected_tokens = selected_tokens.reshape(batch, query_count, block_topk * ratio)
    selected_valid = selected_valid.reshape(batch, query_count, block_topk * ratio)

    if query_indices is None:
        query_ids = torch.arange(start, end, device=selected.device)
    else:
        query_ids = query_indices[start:end].to(selected.device)
    anchors = metadata.query_anchors.index_select(1, query_ids).to(selected.device)
    query_keep = metadata.query_keep.index_select(1, query_ids).to(selected.device)
    safe_anchors = anchors.clamp(0, max(context_length - 1, 0))
    anchor_offsets = torch.gather(
        layout.context_run_offsets, 1, safe_anchors
    )
    tail_start = anchors - anchor_offsets.remainder(ratio)
    tail_offsets = torch.arange(ratio - 1, device=selected.device).view(1, 1, -1)
    tail = tail_start.unsqueeze(-1) + tail_offsets
    tail_valid = (tail < anchors.unsqueeze(-1)) & query_keep.unsqueeze(-1)
    if metadata.query_docs is not None:
        tail_valid &= (
            metadata.query_docs.index_select(1, query_ids)
            .to(selected.device)
            .unsqueeze(-1)
            >= 0
        )

    owners = torch.div(query_ids, metadata.block_size, rounding_mode="floor")
    draft = context_length + owners.view(1, -1, 1) * metadata.block_size
    draft = draft + torch.arange(metadata.block_size, device=selected.device).view(1, 1, -1)
    draft = draft.expand(batch, -1, -1)
    draft_valid = (draft < context_length + draft_length) & query_keep.unsqueeze(-1)
    return (
        torch.cat((selected_tokens, tail, draft), dim=-1),
        torch.cat((selected_valid, tail_valid, draft_valid), dim=-1),
        block_topk * ratio,
    )


def _selected_teacher_target(
    *,
    main_query: torch.Tensor,
    main_key: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    selected: torch.Tensor,
    start: int,
    end: int,
    scale: float,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Selected-set teacher without a Q x full-context workspace."""

    batch, _, query_heads, head_dim = main_query.shape
    num_kv_heads = main_key.shape[2]
    if query_heads % num_kv_heads:
        raise ValueError("QSA teacher query heads must be divisible by KV heads")
    repetitions = query_heads // num_kv_heads
    tokens, token_valid, selected_width = _selected_raw_tokens(
        metadata=metadata,
        layout=layout,
        selected=selected,
        start=start,
        end=end,
        draft_length=main_key.shape[1] - layout.context_group_ids.shape[1],
        query_indices=query_indices,
    )
    safe_tokens = tokens.clamp_min(0)
    probabilities = main_query.new_zeros(tokens.shape, dtype=torch.float32)
    valid_rows = token_valid.any(dim=-1)
    with torch.no_grad():
        grouped_query = main_query[:, start:end].view(
            batch, end - start, num_kv_heads, repetitions, head_dim
        )
        for kv_head in range(num_kv_heads):
            source = main_key[:, :, kv_head]
            source = source[:, None].expand(-1, end - start, -1, -1)
            gathered = torch.gather(
                source,
                2,
                safe_tokens.unsqueeze(-1).expand(-1, -1, -1, head_dim),
            )
            scores = torch.einsum(
                "bqrd,bqkd->bqrk",
                grouped_query[:, :, kv_head].float(),
                gathered.float(),
            ) * scale
            scores = scores.masked_fill(
                ~token_valid.unsqueeze(2), torch.finfo(scores.dtype).min
            )
            scores = torch.where(
                valid_rows.unsqueeze(-1).unsqueeze(-1),
                scores,
                torch.zeros_like(scores),
            )
            probabilities.add_(
                (scores.softmax(dim=-1) * token_valid.unsqueeze(2)).sum(dim=2)
            )
        ratio = layout.token_indices.shape[-1]
        target = probabilities[:, :, :selected_width].view(
            batch, end - start, selected.shape[-1], ratio
        ).max(dim=-1).values
        target = target * selected.ge(0)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _distributed_qsa_token_rows(
    *,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    local_selected: torch.Tensor,
    direct_cp_metadata,
    block_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build rank-local selected raw tokens for every global Draft block."""

    from aspec.models.ops.distributed_sparse import owner_local_rows

    batch, blocks, selected_count = local_selected.shape
    if block_indices is None:
        block_indices = torch.arange(blocks, device=local_selected.device)
    else:
        block_indices = block_indices.to(local_selected.device, dtype=torch.long)
    if block_indices.numel() != blocks:
        raise ValueError("QSA distributed block ids must match selected rows")
    ratio = layout.token_indices.shape[-1]
    if layout.num_groups:
        safe_selected = local_selected.clamp_min(0)
        batch_ids = torch.arange(batch, device=local_selected.device).view(-1, 1, 1)
        selected_tokens = layout.token_indices[batch_ids, safe_selected]
        selected_valid = local_selected.ge(0).unsqueeze(-1) & layout.valid[
            batch_ids, safe_selected
        ].unsqueeze(-1)
        selected_tokens = selected_tokens.reshape(batch, blocks, selected_count * ratio)
        selected_valid = selected_valid.expand(-1, -1, -1, ratio).reshape_as(
            selected_tokens
        )
    else:
        selected_tokens = torch.zeros(
            batch,
            blocks,
            selected_count * ratio,
            dtype=torch.long,
            device=local_selected.device,
        )
        selected_valid = torch.zeros_like(selected_tokens, dtype=torch.bool)

    anchors = metadata.anchor_positions.to(local_selected.device).index_select(
        1, block_indices
    )
    anchor_rope = (
        metadata.anchor_rope_position_ids.to(local_selected.device).index_select(
            1, block_indices
        )
        if metadata.anchor_rope_position_ids is not None
        else anchors
    )
    tail_count = ratio - 1
    tail_offsets = torch.arange(
        tail_count, device=local_selected.device, dtype=torch.long
    ).view(1, 1, -1)
    tail_start = anchors - anchor_rope.remainder(ratio)
    tail_physical = tail_start.unsqueeze(-1) + tail_offsets
    tail_local, tail_owned = owner_local_rows(
        direct_cp_metadata.context_position_ids.to(local_selected.device),
        tail_physical,
    )
    tail_valid = tail_owned & tail_physical.lt(anchors.unsqueeze(-1))
    block_keep = metadata.block_keep_mask.to(local_selected.device).index_select(
        1, block_indices
    )
    tail_valid &= block_keep.unsqueeze(-1)
    if metadata.context_doc_ids is not None and metadata.context_doc_ids.shape[1]:
        local_docs = torch.gather(
            metadata.context_doc_ids.to(local_selected.device),
            1,
            tail_local.flatten(1),
        ).view_as(tail_local)
        anchor_docs = metadata.anchor_doc_ids.to(local_selected.device).index_select(
            1, block_indices
        )
        tail_valid &= local_docs.eq(anchor_docs.unsqueeze(-1))

    local_block_ids = direct_cp_metadata.local_block_ids.to(local_selected.device)
    global_to_local = torch.full(
        (direct_cp_metadata.num_blocks,),
        -1,
        dtype=torch.long,
        device=local_selected.device,
    )
    global_to_local[local_block_ids] = torch.arange(
        local_block_ids.numel(), device=local_selected.device
    )
    local_blocks = global_to_local[block_indices]
    draft_offsets = torch.arange(
        metadata.block_size, device=local_selected.device
    ).view(1, 1, -1)
    draft_tokens = (
        direct_cp_metadata.context_length
        + local_blocks.clamp_min(0).view(1, -1, 1) * metadata.block_size
        + draft_offsets
    ).expand(batch, -1, -1)
    draft_valid = (
        local_blocks.ge(0).view(1, -1, 1)
        & block_keep.unsqueeze(-1)
    ).expand_as(draft_tokens)
    return (
        torch.cat((selected_tokens, tail_local, draft_tokens), dim=-1),
        torch.cat((selected_valid, tail_valid, draft_valid), dim=-1),
        selected_count * ratio,
    )


@torch.no_grad()
def _distributed_selected_teacher_target(
    *,
    main_query: torch.Tensor,
    main_key: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    local_selected: torch.Tensor,
    direct_cp_metadata,
    scale: float,
    workspace_bytes: int,
    block_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute QSA's selected-set teacher without moving selected K/V owners."""

    token_rows, token_valid, selected_width = _distributed_qsa_token_rows(
        metadata=metadata,
        layout=layout,
        local_selected=local_selected,
        direct_cp_metadata=direct_cp_metadata,
        block_indices=block_indices,
    )
    batch, query_count, capacity = token_rows.shape
    query_heads = main_query.shape[2]
    kv_heads = main_key.shape[2]
    if query_heads % kv_heads:
        raise ValueError("QSA teacher query heads must be divisible by KV heads")
    repetitions = query_heads // kv_heads
    ratio = layout.token_indices.shape[-1]
    selected_count = local_selected.shape[-1]
    target = main_query.new_zeros(
        batch, query_count, selected_count, dtype=torch.float32
    )
    # One KV head's gathered keys coexist with the complete FP32
    # [KV, repetition, selected-token] score workspace. Bounding both lets us
    # batch softmax collectives across all KV heads without turning long-context
    # selected-K training into an unbounded allocation.
    bytes_per_query = max(
        batch
        * capacity
        * (
            main_key.shape[-1] * main_key.element_size()
            + (query_heads + repetitions + 2) * torch.float32.itemsize
        ),
        1,
    )
    chunk_size = max(1, workspace_bytes // bytes_per_query)
    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        indices = token_rows[:, start:end]
        valid = token_valid[:, start:end]
        grouped_query = main_query[:, start:end].view(
            batch, end - start, kv_heads, repetitions, main_query.shape[-1]
        )
        scores = target.new_empty(
            batch, end - start, kv_heads, repetitions, capacity
        )
        for kv_head in range(kv_heads):
            source = main_key[:, :, kv_head]
            gathered = torch.gather(
                source[:, None].expand(-1, end - start, -1, -1),
                2,
                indices.unsqueeze(-1).expand(-1, -1, -1, source.shape[-1]),
            )
            head_scores = torch.einsum(
                "bqrd,bqkd->bqrk",
                grouped_query[:, :, kv_head].float(),
                gathered.float(),
            ) * scale
            scores[:, :, kv_head].copy_(
                head_scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
            )
        global_max = scores.amax(dim=-1)
        dist.all_reduce(
            global_max,
            op=dist.ReduceOp.MAX,
            group=direct_cp_metadata.group,
        )
        safe_max = torch.where(
            torch.isfinite(global_max), global_max, torch.zeros_like(global_max)
        )
        scores.sub_(safe_max.unsqueeze(-1)).exp_()
        scores.masked_fill_(~valid[:, :, None, None], 0.0)
        denominator = scores.sum(dim=-1)
        dist.all_reduce(denominator, group=direct_cp_metadata.group)
        scores.div_(denominator.clamp_min(1e-20).unsqueeze(-1))
        token_probability = scores.sum(dim=(2, 3))
        local_target = token_probability[:, :, :selected_width].view(
            batch, end - start, selected_count, ratio
        ).amax(dim=-1)
        dist.all_reduce(local_target, group=direct_cp_metadata.group)
        target[:, start:end] = local_target / local_target.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
    return target


def qsa_distributed_sparse_attention(
    *,
    query_own: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    local_selected: torch.Tensor,
    direct_cp_metadata,
    scale: float,
    workspace_bytes: int,
) -> torch.Tensor:
    """Run QSA sparse core using only local selected K/V plus partial merging."""

    from aspec.models.ops.compact_varlen import distributed_compact_varlen_attention

    token_indices, token_valid, _ = _distributed_qsa_token_rows(
        metadata=metadata,
        layout=layout,
        local_selected=local_selected,
        direct_cp_metadata=direct_cp_metadata,
    )
    output, _ = distributed_compact_varlen_attention(
        query_own=query_own,
        key=key,
        value=value,
        token_indices=token_indices,
        token_valid=token_valid,
        sequence_keep=metadata.block_keep_mask.to(query_own.device),
        queries_per_sequence=metadata.block_size,
        scale=scale,
        workspace_bytes=workspace_bytes,
        group=direct_cp_metadata.group,
    )
    return output


class QSAIndexer(nn.Module):
    """Qwen's compressed MQA indexer with bounded query workspaces."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.qsa_index_n_heads
        self.head_dim = config.qsa_index_head_dim
        self.token_budget = config.qsa_token_budget
        self.ratio = config.qsa_compress_ratio
        self.block_topk = self.token_budget // self.ratio
        self.workspace_bytes = config.qsa_workspace_bytes
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = _PartialRotaryEmbedding(
            config.qsa_rope_dim, config.qsa_rope_theta
        )
        self.scale = self.head_dim**-0.5

    def _project(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        layout: QSACompressionLayout,
        direct_cp_metadata=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.q_proj(draft_hidden).view(
            *draft_hidden.shape[:2], self.num_heads, self.head_dim
        )
        query = self.rotary(self.q_norm(query), draft_position_ids)
        if direct_cp_metadata is not None and direct_cp_metadata.cp_size > 1:
            from aspec.models.ops.direct_cp import all_gather_query_rows

            query = all_gather_query_rows(
                query, group=direct_cp_metadata.group
            )
        token_key = self.k_proj(context_hidden)
        grouped = _gather_groups(token_key, layout)
        compressed = grouped.float().mean(dim=2).to(token_key.dtype)
        compressed = self.rotary(
            self.k_norm(compressed).unsqueeze(2), layout.rope_positions
        ).squeeze(2)
        compressed = torch.where(
            layout.valid.unsqueeze(-1), compressed, torch.zeros_like(compressed)
        )
        return query, compressed

    def _scores(
        self, query: torch.Tensor, compressed_key: torch.Tensor
    ) -> torch.Tensor:
        # Keep the four lightweight MQA heads in one batched GEMM.  A Python
        # head loop creates four independent backward graphs and is visibly
        # slower at the 256K-CPT shapes QSA is designed for.
        return (
            torch.einsum(
                "bqhd,bkd->bqhk", query.float(), compressed_key.float()
            )
            .relu_()
            .sum(dim=2)
            * self.scale
        )

    def forward(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        metadata: QSAForwardMetadata,
        layout: QSACompressionLayout,
        main_query: torch.Tensor,
        main_key: torch.Tensor,
        stage: str,
        main_scale: float,
        query_indices: torch.Tensor | None = None,
        direct_cp_metadata=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        query, compressed_key = self._project(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            layout=layout,
            direct_cp_metadata=direct_cp_metadata,
        )
        local_query_count = draft_hidden.shape[1]
        distributed = (
            direct_cp_metadata is not None and direct_cp_metadata.cp_size > 1
        )
        if distributed:
            from aspec.models.ops.direct_cp import all_gather_query_rows

            main_query = all_gather_query_rows(
                main_query, group=direct_cp_metadata.group
            )
        query_count = query.shape[1]
        from aspec.models.ops.distributed_sparse import max_context_shard_length

        context_capacity = max_context_shard_length(
            direct_cp_metadata, context_hidden.shape[1], alignment=self.ratio
        )
        group_capacity = context_capacity // self.ratio
        if stage == "indexer_warmup":
            teacher_key_equivalent = (
                context_capacity + main_key.shape[1] - context_hidden.shape[1]
            )
            # The distributed teacher batches all KV heads into one bounded
            # score workspace so its softmax needs only two collectives per
            # query chunk. Account for that workspace before entering it.
            repetitions = main_query.shape[2] // main_key.shape[2]
            workspace_multiplier = main_query.shape[2] + repetitions + 2
        else:
            selected_token_capacity = (
                min(self.token_budget, group_capacity * self.ratio)
                + self.ratio
                - 1
                + metadata.block_size
            )
            # Selected-key gather is BF16/FP16, while the workspace helper is
            # expressed in FP32 scalar slots. Account for the head dimension
            # before choosing a chunk; otherwise a 128K sparse step can retain
            # multi-GiB [Q, selected_tokens, head_dim] tensors.
            teacher_key_equivalent = math.ceil(
                selected_token_capacity * main_query.shape[-1] / 2
            )
            workspace_multiplier = 2
        chunk_size = min(
            query_count,
            _query_chunk_size(
                batch=query.shape[0],
                keys=max(group_capacity, teacher_key_equivalent),
                workspace_bytes=self.workspace_bytes,
                multiplier=workspace_multiplier,
            ),
        )
        selections: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        topk = self.block_topk if distributed else min(self.block_topk, layout.num_groups)
        for start in range(0, query_count, chunk_size):
            end = min(start + chunk_size, query_count)
            eligible = qsa_eligible_groups(
                metadata, layout, start, end, query_indices=query_indices
            )
            scores = self._scores(query[:, start:end], compressed_key)
            scores = scores.masked_fill(~eligible, -torch.inf)
            valid = eligible.any(dim=-1)
            scores = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
            if stage == "indexer_warmup":
                if direct_cp_metadata is None:
                    target = _full_teacher_target(
                        main_query=main_query.detach(),
                        main_key=main_key.detach(),
                        metadata=metadata,
                        layout=layout,
                        start=start,
                        end=end,
                        scale=main_scale,
                        query_indices=query_indices,
                    )
                    log_prob, valid = _safe_log_softmax(scores, eligible)
                    loss = F.kl_div(
                        log_prob, target, reduction="none"
                    ).sum(dim=-1)
                    losses.append(
                        torch.where(valid, loss, torch.zeros_like(loss))
                    )
                    continue
                from aspec.models.ops.distributed_sparse import (
                    distributed_log_softmax,
                )

                log_prob, valid = distributed_log_softmax(
                    scores,
                    eligible,
                    group=direct_cp_metadata.group,
                )
                target, teacher_valid = _distributed_full_teacher_target(
                    main_query=main_query[:, start:end].detach(),
                    main_key=main_key.detach(),
                    metadata=metadata,
                    layout=layout,
                    query_indices=query_indices[start:end],
                    direct_cp_metadata=direct_cp_metadata,
                    scale=main_scale,
                )
                local_loss = F.kl_div(
                    log_prob, target, reduction="none"
                ).sum(dim=-1)
                if direct_cp_metadata.cp_size > 1:
                    from torch.distributed.nn.functional import all_reduce

                    loss = all_reduce(
                        local_loss, group=direct_cp_metadata.group
                    )
                else:
                    loss = local_loss
                losses.append(
                    torch.where(
                        valid & teacher_valid,
                        loss,
                        torch.zeros_like(loss),
                    )
                )
                continue
            if distributed:
                from aspec.models.ops.distributed_sparse import distributed_topk

                winners = distributed_topk(
                    scores,
                    eligible,
                    topk=topk,
                    group=direct_cp_metadata.group,
                )
                selected = winners.local_winner_indices
                selected_scores = winners.scores
                selected_valid = winners.valid
            elif topk == 0:
                selected = torch.empty(
                    *scores.shape[:2], 0, dtype=torch.long, device=scores.device
                )
            else:
                selected = scores.topk(topk, dim=-1, sorted=False).indices
                selected_valid = torch.gather(eligible, -1, selected)
                selected = torch.where(
                    selected_valid, selected, torch.full_like(selected, -1)
                )
            selections.append(selected)
            if not getattr(self, "compute_auxiliary_loss", True):
                losses.append(scores.new_zeros(scores.shape[:2]))
                continue
            if not distributed:
                selected_valid = selected >= 0
                selected_scores = torch.gather(scores, -1, selected.clamp_min(0))
                target = _selected_teacher_target(
                    main_query=main_query.detach(),
                    main_key=main_key.detach(),
                    metadata=metadata,
                    layout=layout,
                    selected=selected,
                    start=start,
                    end=end,
                    scale=main_scale,
                    query_indices=query_indices,
                )
            else:
                target = _distributed_selected_teacher_target(
                    main_query=main_query[:, start:end].detach(),
                    main_key=main_key.detach(),
                    metadata=metadata,
                    layout=layout,
                    local_selected=selected,
                    direct_cp_metadata=direct_cp_metadata,
                    scale=main_scale,
                    workspace_bytes=self.workspace_bytes,
                    block_indices=torch.div(
                        query_indices[start:end],
                        metadata.block_size,
                        rounding_mode="floor",
                    ),
                )
            log_prob, valid = _safe_log_softmax(selected_scores, selected_valid)
            loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)
            loss = torch.where(valid, loss, torch.zeros_like(loss))
            losses.append(loss)
        selected = torch.cat(selections, dim=1) if selections else None
        loss = (
            torch.cat(losses, dim=1)
            if losses
            else main_query.new_zeros(main_query.shape[:2], dtype=torch.float32)
        )
        if distributed:
            rank = dist.get_rank(direct_cp_metadata.group)
            local_start = rank * local_query_count
            loss = loss[:, local_start : local_start + local_query_count]
        return selected, loss


def _selection_words(selected: torch.Tensor, num_groups: int) -> torch.Tensor:
    word_count = math.ceil(num_groups / 32)
    words = torch.zeros(
        *selected.shape[:2], word_count, dtype=torch.int64, device=selected.device
    )
    if word_count == 0 or selected.shape[-1] == 0:
        return words
    valid = selected >= 0
    safe = selected.clamp_min(0)
    word_ids = torch.div(safe, 32, rounding_mode="floor")
    bits = torch.bitwise_left_shift(
        torch.ones_like(safe, dtype=torch.int64), safe.remainder(32)
    )
    words.scatter_add_(2, word_ids, torch.where(valid, bits, torch.zeros_like(bits)))
    return words


def build_qsa_reference_mask_mod(
    *,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    selected: torch.Tensor,
    context_length: int,
    draft_length: int,
    device: torch.device,
):
    """Return the exact QSA predicate used by CPU/reference checks.

    Production sparse attention deliberately does not build a FlexAttention
    ``BlockMask``.  Random QSA groups make coarse block metadata over-read most
    tiles at long context; the production path physically gathers the selected
    raw K/V once per Draft block and calls FA4 varlen attention instead.
    """

    query_length = selected.shape[1]
    total_length = context_length + draft_length
    words = _selection_words(selected, layout.num_groups)
    anchors = metadata.query_anchors.to(device)
    keep = metadata.query_keep.to(device)
    query_docs = metadata.query_docs
    if query_docs is not None:
        query_docs = query_docs.to(device)
        context_docs = metadata.context_doc_ids.to(device)
    else:
        context_docs = None
    safe_anchors = anchors.clamp(0, max(context_length - 1, 0))
    anchor_offsets = torch.gather(layout.context_run_offsets, 1, safe_anchors)
    tail_starts = anchors - anchor_offsets.remainder(layout.token_indices.shape[-1])
    group_ids = layout.context_group_ids

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        is_context = kv_idx < context_length
        safe_context = kv_idx.clamp_max(context_length - 1)
        group = group_ids[b, safe_context]
        if words.shape[-1]:
            safe_group = torch.maximum(group, torch.zeros_like(group))
            word = words[b, q_idx, safe_group // 32]
            selected_group = (group >= 0) & (((word >> (safe_group % 32)) & 1) != 0)
        else:
            selected_group = group < -1
        tail = is_context & (kv_idx >= tail_starts[b, q_idx]) & (
            kv_idx < anchors[b, q_idx]
        )
        if context_docs is not None:
            same_doc = (query_docs[b, q_idx] >= 0) & (
                context_docs[b, safe_context] == query_docs[b, q_idx]
            )
            selected_group = selected_group & same_doc
            tail = tail & same_doc
        owner = q_idx // metadata.block_size
        draft_owner = (kv_idx - context_length) // metadata.block_size
        draft = (kv_idx >= context_length) & (kv_idx < total_length) & (
            draft_owner == owner
        )
        return keep[b, q_idx] & ((is_context & selected_group) | tail | draft)

    mask_mod.__name__ = (
        f"dflash_qsa_Q{query_length}_K{total_length}_G{layout.num_groups}_"
        f"T{selected.shape[-1]}"
    )

    return mask_mod


def qsa_sparse_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: QSAForwardMetadata,
    layout: QSACompressionLayout,
    selected: torch.Tensor,
    scale: float,
    workspace_bytes: int,
) -> torch.Tensor:
    """Run exact selected-token attention through bounded varlen gathers.

    A generic block-sparse kernel is a poor fit for QSA's random four-token
    micro-blocks: the union of unrelated selections makes nearly every 128x128
    tile partial.  Qwen MTP shares one selection inside a Draft block, so gather
    each block's compact raw K/V once and execute a normal FlashAttention varlen
    sequence.  Checkpointing recomputes the bounded gather during backward
    instead of retaining one [anchors, K, heads, dim] tensor per Draft layer.
    """

    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise NotImplementedError("QSA sparse core requires CUDA FP16/BF16")
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except ImportError as error:
        raise NotImplementedError(
            "QSA sparse core requires FlashAttention-4 varlen support"
        ) from error
    from torch.utils.checkpoint import checkpoint

    batch, query_length, query_heads, head_dim = query.shape
    if key.shape[:2] != value.shape[:2] or key.shape[2:] != value.shape[2:]:
        raise ValueError("QSA key/value tensors must have matching BSHD shapes")
    if key.shape[0] != batch or query_heads % key.shape[2]:
        raise ValueError("QSA query heads must be divisible by key/value heads")
    if query_length != metadata.anchor_positions.shape[1] * metadata.block_size:
        raise ValueError(
            "QSA query length must equal anchors times Draft block size: "
            f"{query_length} != {metadata.anchor_positions.shape[1]} * "
            f"{metadata.block_size}"
        )
    anchors = metadata.anchor_positions.shape[1]
    if selected.ndim != 3 or selected.shape[:2] != (batch, anchors):
        raise ValueError(
            "QSA sparse selection must contain one compact Top-K row per Draft "
            f"block: expected [{batch}, {anchors}, topk], got {list(selected.shape)}"
        )

    representative_rows = torch.arange(
        0, query_length, metadata.block_size, device=query.device
    )
    tokens, token_valid, _ = _selected_raw_tokens(
        metadata=metadata,
        layout=layout,
        selected=selected,
        start=0,
        end=representative_rows.numel(),
        draft_length=key.shape[1] - layout.context_group_ids.shape[1],
        query_indices=representative_rows,
    )
    active = metadata.block_keep_mask.bool().reshape(-1)
    active_rows = active.nonzero(as_tuple=False).flatten()
    output_rows = query.new_zeros(
        batch * anchors, metadata.block_size, query_heads, head_dim
    )
    if active_rows.numel() == 0:
        return output_rows.view(batch, query_length, query_heads, head_dim)

    query_rows = query.view(
        batch * anchors, metadata.block_size, query_heads, head_dim
    ).index_select(0, active_rows)
    token_rows = tokens.view(batch * anchors, -1).index_select(0, active_rows)
    valid_rows = token_valid.view(batch * anchors, -1).index_select(0, active_rows)
    row_batches = (
        torch.arange(batch, device=query.device)
        .view(-1, 1)
        .expand(-1, anchors)
        .reshape(-1)
        .index_select(0, active_rows)
    )
    selected_capacity = token_rows.shape[1]
    bytes_per_row = max(
        selected_capacity
        * key.shape[2]
        * head_dim
        * key.element_size()
        * 2,
        1,
    )
    rows_per_chunk = max(1, workspace_bytes // bytes_per_row)
    chunks = []
    for start in range(0, active_rows.numel(), rows_per_chunk):
        end = min(start + rows_per_chunk, active_rows.numel())
        q_rows = query_rows[start:end]
        token_chunk = token_rows[start:end]
        valid_chunk = valid_rows[start:end]
        batch_chunk = row_batches[start:end]
        def attend(
            q_input,
            k_input,
            v_input,
            token_input,
            valid_input,
            batch_input,
            # FA4 accepts an upper bound.  Keep it on the host-side shape
            # contract instead of calling ``.item()`` on a CUDA reduction for
            # every workspace chunk, which would serialize the launch stream.
            _max_key_length=selected_capacity,
        ):
            flat_valid = valid_input.reshape(-1)
            flat_tokens = token_input.reshape(-1)[flat_valid]
            flat_batches = (
                batch_input[:, None]
                .expand_as(token_input)
                .reshape(-1)[flat_valid]
            )
            gathered_key = k_input[flat_batches, flat_tokens].contiguous()
            gathered_value = v_input[flat_batches, flat_tokens].contiguous()
            q_flat = q_input.reshape(-1, query_heads, head_dim).contiguous()
            counts = valid_input.sum(dim=-1, dtype=torch.int32)
            cu_query = torch.arange(
                0,
                (q_input.shape[0] + 1) * metadata.block_size,
                metadata.block_size,
                dtype=torch.int32,
                device=q_input.device,
            )
            cu_key = torch.cat(
                (counts.new_zeros(1), counts.cumsum(dim=0, dtype=torch.int32)),
                dim=0,
            )
            output, _ = flash_attn_varlen_func(
                q_flat,
                gathered_key,
                gathered_value,
                cu_seqlens_q=cu_query,
                cu_seqlens_k=cu_key,
                max_seqlen_q=metadata.block_size,
                max_seqlen_k=_max_key_length,
                softmax_scale=scale,
                causal=False,
                pack_gqa=query_heads != key.shape[2],
            )
            return output.view(q_input.shape)

        chunks.append(
            checkpoint(
                attend,
                q_rows,
                key,
                value,
                token_chunk,
                valid_chunk,
                batch_chunk,
                use_reentrant=False,
            )
        )
    active_output = torch.cat(chunks, dim=0)
    output_rows = output_rows.index_copy(0, active_rows, active_output)
    return output_rows.view(batch, query_length, query_heads, head_dim)


class QSALayer(nn.Module):
    """Per-layer QSA selector; compressed keys never enter core attention."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.stage = config.qsa_training_stage
        self.indexer_loss_coef = config.qsa_indexer_loss_coef
        self.indexer = QSAIndexer(config)

    def configure_trainable_parameters(self) -> None:
        if self.stage != "indexer_warmup":
            return
        for parameter in self.indexer.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        *,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        metadata: QSAForwardMetadata,
        layout: QSACompressionLayout | None,
        main_query: torch.Tensor,
        main_key: torch.Tensor,
        scale: float,
        direct_cp_metadata=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict]:
        if self.stage == "dense" or (self.stage == "indexer_warmup" and not getattr(self, "compute_auxiliary_loss", True)):
            loss = main_query.new_zeros(main_query.shape[:2], dtype=torch.float32)
            selected = None
        else:
            if layout is None:
                raise ValueError("QSA indexer stages require a compression layout")
            local_query_indices = torch.arange(
                0,
                draft_hidden.shape[1],
                metadata.block_size,
                device=draft_hidden.device,
            )
            distributed = (
                direct_cp_metadata is not None and direct_cp_metadata.cp_size > 1
            )
            expected_blocks = (
                direct_cp_metadata.blocks_per_rank
                if distributed else metadata.anchor_positions.shape[1]
            )
            if local_query_indices.numel() != expected_blocks:
                raise ValueError(
                    "QSA Draft queries must contain one complete block per anchor"
                )
            query_indices = (
                direct_cp_metadata.collective_block_ids.to(draft_hidden.device)
                * metadata.block_size
                if distributed else local_query_indices
            )
            selected, loss = self.indexer(
                draft_hidden=draft_hidden.index_select(1, local_query_indices),
                context_hidden=context_hidden,
                draft_position_ids=draft_position_ids.index_select(
                    1, local_query_indices
                ),
                metadata=metadata,
                layout=layout,
                main_query=main_query.index_select(1, local_query_indices),
                main_key=main_key,
                stage=self.stage,
                main_scale=scale,
                query_indices=query_indices,
                direct_cp_metadata=direct_cp_metadata,
            )
            # Keep one Top-K row per Draft block. Every query in the block uses
            # the same selection, so expanding it by block_size only wastes
            # device memory and bandwidth before the sparse core compacts it
            # again. The per-query KL is still repeated for objective weights.
            loss = loss.repeat_interleave(metadata.block_size, dim=1)
        return selected, loss, {
            "layer": self.layer_idx,
            "type": "qwen_sparse_attention",
            "variant": "qsa",
            "training_stage": self.stage,
            "compress_ratio": self.indexer.ratio,
            "token_budget": self.indexer.token_budget,
            "block_topk": self.indexer.block_topk,
            "index_n_heads": self.indexer.num_heads,
            "index_head_dim": self.indexer.head_dim,
            "rope_dim": self.indexer.rotary.rotary_dim,
            "indexer_input_detached": False,
            "cross_layer_reuse": False,
            "draft_block_index_reuse": True,
            "sparse_core_backend": "flash_attn_varlen_gather",
        }


__all__ = [
    "QSACompressionLayout",
    "QSAForwardMetadata",
    "QSAIndexer",
    "QSALayer",
    "build_qsa_reference_mask_mod",
    "build_qsa_compression_layout",
    "qsa_sparse_attention",
    "qsa_eligible_groups",
]
