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

"""Token-granular DeepSeek Sparse Attention for DSpark/DFlash Drafts.

Reimplemented from DeepSeek-V3.2's public indexer/continued-training semantics
(https://github.com/deepseek-ai/DeepSeek-V3.2-Exp and arXiv:2512.02556).
The GQA Draft owns its query adapter; it does not pretend to share an MLA latent.
Existing ASpec token-selection, teacher and owner-local FA4 primitives are
reused, without QSA compression or its block-shared index approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aspec.models.draft.qsa import (
    QSACompressionLayout,
    QSAForwardMetadata,
    _distributed_full_teacher_target,
    _distributed_qsa_token_rows,
    _distributed_selected_teacher_target,
    _safe_log_softmax,
    _selected_raw_tokens,
    _selected_teacher_target,
    qsa_eligible_groups,
)

# Visibility metadata has the same physical/document coordinate contract.
DSAForwardMetadata = QSAForwardMetadata


def _rotary_factors(positions, dim, theta):
    inverse = theta ** (-torch.arange(0, dim, 2, device=positions.device).float() / dim)
    angles = positions.float().unsqueeze(-1) * inverse
    return angles.cos(), angles.sin()


def _apply_rotary(values, factors):
    """DeepSeek Indexer's non-interleaved RoPE, unlike its main MLA path."""
    cos, sin = factors
    dim = cos.shape[-1] * 2
    first, second = values[..., :dim].float().chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return torch.cat((rotated.to(values.dtype), values[..., dim:]), dim=-1)


class _RotaryFactors:
    """One forward's FP32 factors, opaque to FSDP's recursive input casting.

    Otherwise every layer copies/rounds these shared factors to BF16, changing
    the Indexer's rotation while also allocating a context-sized cache copy.
    """

    def __init__(self, query, key):
        self.query, self.key = query, key


@dataclass(frozen=True)
class DSATokenLayout(QSACompressionLayout):
    """Identity token rows, never compressed; one layout per model forward."""

    rotary_factors: _RotaryFactors

    @property
    def query_rotary(self):
        return self.rotary_factors.query

    @property
    def key_rotary(self):
        return self.rotary_factors.key


def build_dsa_token_layout(metadata, context_positions, draft_positions, *, rope_dim, rope_theta):
    batch, length = context_positions.shape
    rows = torch.arange(length, device=context_positions.device).expand(batch, -1)
    physical = metadata.context_physical_position_ids
    physical = rows if physical is None else physical.to(rows.device)
    docs = metadata.context_doc_ids
    docs = torch.zeros_like(rows) if docs is None else docs.to(rows.device)
    if physical.shape != rows.shape or docs.shape != rows.shape:
        raise ValueError("DSA context positions and document metadata must have equal shapes")
    valid = docs >= 0
    return DSATokenLayout(
        token_indices=rows.unsqueeze(-1), valid=valid,
        physical_starts=physical, physical_ends=physical + 1,
        rope_positions=context_positions, doc_ids=docs,
        context_group_ids=torch.where(valid, rows, -1),
        # Identity groups have no incomplete compression tail.
        context_run_offsets=torch.zeros_like(rows),
        rotary_factors=_RotaryFactors(
            _rotary_factors(draft_positions, rope_dim, rope_theta),
            _rotary_factors(context_positions, rope_dim, rope_theta),
        ),
    )


def _batched_dot(query, key):
    """One trainable Tensor Core GEMM, followed by FP32 score arithmetic."""
    batch, rows, heads, width = query.shape
    left = query.reshape(batch, rows * heads, width)
    right = key.transpose(1, 2)
    if query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}:
        # out_dtype=float32 has no autograd implementation in the pinned
        # PyTorch lane. Normal mixed-precision bmm keeps both tensor-core
        # forward/backward; ReLU, weights and softmax remain FP32.
        scores = torch.bmm(left, right).float()
    else:
        scores = torch.bmm(left.float(), right.float())
    return scores.view(batch, rows, heads, key.shape[1])


def _owner_geometry(key, metadata):
    """One forward's common search geometry, including uneven CP shards.

    Collective chunk boundaries and candidate widths must not depend on this
    rank's local length. Cache only integer geometry, never activations.
    """
    if metadata is None or metadata.cp_size == 1:
        return key.shape[1], key.shape[1]
    cache_key = ("dsa_owner_geometry",)
    cached = metadata._fa4_cache.get(cache_key)
    if cached is not None:
        return cached
    if metadata.sequence_length is not None:
        from aspec.inference.context import context_shard_lengths

        divisor = metadata.cp_size * (2 if metadata.layout == "zigzag" else 1)
        lengths = context_shard_lengths(metadata.sequence_length, divisor)
        if metadata.layout == "zigzag":
            lengths = [lengths[rank] + lengths[-rank - 1] for rank in range(metadata.cp_size)]
    else:
        # Explicit metadata without global length still works. One small
        # geometry exchange/sync per forward, not per layer or query chunk.
        local = torch.tensor([key.shape[1]], device=key.device, dtype=torch.long)
        gathered = [torch.empty_like(local) for _ in range(metadata.cp_size)]
        dist.all_gather(gathered, local, group=metadata.group)
        lengths = torch.cat(gathered).tolist()
    result = sum(lengths), max(lengths)
    metadata._fa4_cache[cache_key] = result
    return result


class DSALightningIndexer(nn.Module):
    """Weighted-ReLU token indexer with checkpointed, bounded score chunks."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.dsa_index_n_heads
        self.head_dim = config.dsa_index_head_dim
        self.token_budget = config.dsa_token_budget
        self.workspace_bytes = config.dsa_workspace_bytes
        self.q_a_proj = nn.Linear(config.hidden_size, config.dsa_q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(config.dsa_q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.dsa_q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.scale = self.head_dim**-0.5 * self.num_heads**-0.5

    def project(self, draft_hidden, context_hidden, layout):
        draft, context = draft_hidden.detach(), context_hidden.detach()
        query = self.q_b_proj(self.q_a_norm(self.q_a_proj(draft))).unflatten(
            -1, (self.num_heads, self.head_dim)
        )
        key = self.k_norm(self.k_proj(context)).unsqueeze(2)
        query = _apply_rotary(query, layout.query_rotary)
        key = _apply_rotary(key, layout.key_rotary).squeeze(2)
        weights = F.linear(draft.float(), self.weights_proj.weight.float()) * self.scale
        return query, key, weights

    @staticmethod
    def scores(query, key, weights):
        return (_batched_dot(query, key).relu_() * weights.unsqueeze(-1)).sum(dim=2)

    @staticmethod
    def selected_scores(query, key, weights, selected):
        batch_ids = torch.arange(key.shape[0], device=key.device).view(-1, 1, 1)
        gathered = key[batch_ids, selected.clamp_min(0)]
        # [B,Q,H,D] x [B,Q,K,D], not a Q x full-context backward graph.
        batch, rows, heads, width = query.shape
        scores = _batched_dot(
            query.reshape(batch * rows, 1, heads, width),
            gathered.reshape(batch * rows, selected.shape[-1], width),
        ).view(batch, rows, heads, selected.shape[-1])
        return (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    def forward(self, *, draft_hidden, context_hidden, metadata, layout,
                main_query, main_key, stage, main_scale, direct_cp_metadata=None):
        query, key, weights = self.project(draft_hidden, context_hidden, layout)
        local_queries = query.shape[1]
        distributed = direct_cp_metadata is not None and direct_cp_metadata.cp_size > 1
        query_ids = torch.arange(local_queries, device=query.device)
        if direct_cp_metadata is not None:
            query_ids = (
                direct_cp_metadata.collective_block_ids.to(query.device)[:, None] * metadata.block_size
                + torch.arange(metadata.block_size, device=query.device)
            ).flatten()
        if distributed:
            from aspec.models.ops.direct_cp import all_gather_query_rows

            query = all_gather_query_rows(query, group=direct_cp_metadata.group)
            weights = all_gather_query_rows(
                weights.unsqueeze(-1), group=direct_cp_metadata.group
            ).squeeze(-1)
            main_query = all_gather_query_rows(main_query.detach(), group=direct_cp_metadata.group)
        main_query, main_key = main_query.detach(), main_key.detach()
        # Include heads, masks, FP32 loss intermediates and selected teacher gathers.
        # Query chunks alone are insufficient if their score graphs are all retained.
        keys = key.shape[1]
        global_keys, search_keys = _owner_geometry(key, direct_cp_metadata)
        repetitions = main_query.shape[2] // main_key.shape[2]
        # Warmup batches every main-attention head into one bounded teacher
        # score tensor, trading workspace for a fixed number of CP collectives.
        teacher_slots = (
            main_query.shape[2] + repetitions + 2
            if stage == "indexer_warmup"
            else 4
        )
        teacher_keys = search_keys + main_key.shape[1] - keys if stage == "indexer_warmup" else search_keys
        topk = min(self.token_budget, global_keys)
        per_query = max(
            4 * (max(search_keys, 1) * self.num_heads * 2 + max(teacher_keys, 1) * teacher_slots),
            (topk + metadata.block_size) * (
                self.head_dim * 4 + main_query.shape[-1] * 8 + self.num_heads * 12
            ),
        ) * query.shape[0]
        chunk = max(1, self.workspace_bytes // per_query)
        selections, selection_validity, losses = [], [], []
        for start in range(0, query.shape[1], chunk):
            end = min(start + chunk, query.shape[1])
            ids = query_ids[start:end]
            q, w = query[:, start:end], weights[:, start:end]
            if stage == "indexer_warmup":
                def warmup(q_input, k_input, w_input, teacher_q, teacher_k,
                           _ids=ids):
                    mask = qsa_eligible_groups(metadata, layout, query_indices=_ids)
                    scores = self.scores(q_input, k_input, w_input)
                    if distributed:
                        from aspec.models.ops.distributed_sparse import distributed_log_softmax

                        log_prob, valid = distributed_log_softmax(
                            scores, mask, group=direct_cp_metadata.group
                        )
                        target, teacher_valid = _distributed_full_teacher_target(
                            main_query=teacher_q, main_key=teacher_k, metadata=metadata,
                            layout=layout, query_indices=_ids,
                            direct_cp_metadata=direct_cp_metadata, scale=main_scale,
                        )
                        loss = F.kl_div(log_prob, target, reduction="none").sum(-1)
                        from torch.distributed.nn.functional import all_reduce

                        loss = all_reduce(loss, group=direct_cp_metadata.group)
                        valid = valid & teacher_valid
                    else:
                        log_prob, valid = _safe_log_softmax(scores, mask)
                        # The owner-local teacher also works at CP1. It batches
                        # GQA query heads per KV head rather than launching one
                        # separate full-context GEMM for every query head.
                        owner = direct_cp_metadata or SimpleNamespace(
                            context_position_ids=layout.physical_starts,
                            local_block_ids=torch.arange(
                                metadata.anchor_positions.shape[1], device=q_input.device
                            ),
                            context_length=key.shape[1], cp_size=1, group=None,
                        )
                        target, _ = _distributed_full_teacher_target(
                            main_query=teacher_q, main_key=teacher_k, metadata=metadata,
                            layout=layout, query_indices=_ids,
                            direct_cp_metadata=owner, scale=main_scale,
                        )
                        loss = F.kl_div(log_prob, target, reduction="none").sum(-1)
                    return torch.where(valid, loss, torch.zeros_like(loss))

                # Teacher scores and full indexer logits are recomputed, not saved.
                losses.append(checkpoint(
                    warmup, q, key, w, main_query[:, start:end], main_key,
                    use_reentrant=False,
                ))
                continue
            with torch.no_grad():
                eligible = qsa_eligible_groups(metadata, layout, query_indices=ids)
                scores = self.scores(q, key, w)
                if distributed:
                    from aspec.models.ops.distributed_sparse import distributed_topk

                    winners = distributed_topk(scores, eligible, topk=topk,
                                               group=direct_cp_metadata.group)
                    selected = winners.local_winner_indices
                    selected_valid = winners.valid
                else:
                    selected = scores.masked_fill(~eligible, -torch.inf).topk(
                        topk, dim=-1, sorted=False
                    ).indices
                    selected_valid = eligible.gather(-1, selected)
                    selected = torch.where(selected_valid, selected, -1)
            selections.append(selected)
            selection_validity.append(selected_valid)
        selected = torch.cat(selections, dim=1) if selections else None
        selected_validity = (
            torch.cat(selection_validity, dim=1) if selection_validity else None
        )
        if selected is not None and not getattr(self, "compute_auxiliary_loss", True):
            loss = query.new_zeros(query.shape[:2], dtype=torch.float32)
        elif selected is not None:
            # Search sees N context tokens, whereas teacher/KL sees only K
            # winners. Do not inherit N's tiny chunks for K-sized work. Keep
            # the detached K-sized target once, rather than recomputing teacher
            # GEMMs and CP reductions in checkpoint backward.
            per_selected_query = query.shape[0] * (topk + metadata.block_size) * (
                self.head_dim * 4 + main_query.shape[-1] * 8
                + self.num_heads * 12 + repetitions * 12
            )
            selected_chunk = max(1, self.workspace_bytes // max(per_selected_query, 1))
            for start in range(0, query.shape[1], selected_chunk):
                end = min(start + selected_chunk, query.shape[1])
                ids, chosen = query_ids[start:end], selected[:, start:end]
                with torch.no_grad():
                    # distributed_topk already reports global winner validity.
                    # Retaining that K-sized mask avoids another CP collective
                    # for every selected-teacher chunk.
                    valid_selected = selected_validity[:, start:end]
                    if distributed:
                        target = _distributed_selected_teacher_target(
                            main_query=main_query[:, start:end], main_key=main_key,
                            metadata=metadata, layout=layout, local_selected=chosen,
                            direct_cp_metadata=direct_cp_metadata, scale=main_scale,
                            workspace_bytes=self.workspace_bytes,
                            block_indices=ids.div(metadata.block_size, rounding_mode="floor"),
                        )
                    else:
                        target = _selected_teacher_target(
                            main_query=main_query[:, start:end], main_key=main_key,
                            metadata=metadata, layout=layout, selected=chosen,
                            start=0, end=ids.numel(), scale=main_scale, query_indices=ids,
                        )

                def sparse_loss(q_input, k_input, w_input, indices, teacher, valid):
                    selected_logits = self.selected_scores(q_input, k_input, w_input, indices)
                    if distributed:
                        # Every winner slot has exactly one owner. Never gather
                        # context K/V or full-context scores.
                        from torch.distributed.nn.functional import all_reduce

                        selected_logits = all_reduce(
                            torch.where(indices >= 0, selected_logits, 0.0),
                            group=direct_cp_metadata.group,
                        )
                    log_prob, row_valid = _safe_log_softmax(selected_logits, valid)
                    loss = F.kl_div(log_prob, teacher, reduction="none").sum(-1)
                    return torch.where(row_valid, loss, torch.zeros_like(loss))

                losses.append(checkpoint(
                    sparse_loss, query[:, start:end], key, weights[:, start:end],
                    chosen, target, valid_selected, use_reentrant=False,
                ))
            loss = torch.cat(losses, dim=1)
        else:
            loss = torch.cat(losses, dim=1)
        if distributed:
            rank = dist.get_rank(direct_cp_metadata.group)
            loss = loss[:, rank * local_queries:(rank + 1) * local_queries]
        return selected, loss


def dsa_sparse_attention(*, query, key, value, metadata, layout, selected,
                         scale, workspace_bytes, direct_cp_metadata=None, observation=None):
    """Per-query raw-token attention; no block-shared selection or context gather."""
    if direct_cp_metadata is not None:
        from aspec.models.ops.compact_varlen import distributed_compact_varlen_attention

        block_ids = direct_cp_metadata.collective_block_ids.to(query.device)
        block_ids = block_ids.repeat_interleave(metadata.block_size)
        rows, valid, _ = _distributed_qsa_token_rows(
            metadata=metadata, layout=layout, local_selected=selected,
            direct_cp_metadata=direct_cp_metadata, block_indices=block_ids,
        )
        output, _ = distributed_compact_varlen_attention(
            query_own=query, key=key, value=value, token_indices=rows, token_valid=valid,
            sequence_keep=metadata.block_keep_mask.index_select(
                1, direct_cp_metadata.collective_block_ids.to(query.device)
            ).repeat_interleave(metadata.block_size, dim=1),
            queries_per_sequence=1, scale=scale, workspace_bytes=workspace_bytes,
            group=direct_cp_metadata.group,
            observation=observation,
        )
        return output
    from aspec.models.ops.compact_varlen import compact_varlen_attention

    rows, valid, _ = _selected_raw_tokens(
        metadata=metadata, layout=layout, selected=selected,
        start=0, end=query.shape[1], draft_length=query.shape[1],
    )
    return compact_varlen_attention(
        query=query, key=key, value=value, token_indices=rows, token_valid=valid,
        sequence_keep=metadata.query_keep, queries_per_sequence=1,
        scale=scale, workspace_bytes=workspace_bytes,
        observation=observation,
    )


class DSALayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.stage = config.dsa_training_stage
        self.indexer = DSALightningIndexer(config)
        self.indexer.compute_auxiliary_loss = config.dsa_indexer_loss_coef > 0

    def forward(self, *, draft_hidden, context_hidden, draft_position_ids,
                metadata, layout, main_query, main_key, scale, direct_cp_metadata=None):
        if self.stage == "dense" or (
            self.stage == "indexer_warmup" and not getattr(self, "compute_auxiliary_loss", True)
        ):
            selected, loss = None, main_query.new_zeros(main_query.shape[:2], dtype=torch.float32)
        else:
            if layout is None:
                raise ValueError("DSA indexer stages require a token layout")
            selected, loss = self.indexer(
                draft_hidden=draft_hidden, context_hidden=context_hidden,
                metadata=metadata, layout=layout, main_query=main_query, main_key=main_key,
                stage=self.stage, main_scale=scale, direct_cp_metadata=direct_cp_metadata,
            )
        return selected, loss, {
            "layer": self.layer_idx, "type": "deepseek_sparse_attention", "variant": "dsa",
            "training_stage": self.stage, "token_budget": self.indexer.token_budget,
            "index_n_heads": self.indexer.num_heads, "index_head_dim": self.indexer.head_dim,
            "q_lora_rank": self.indexer.q_a_proj.out_features,
            "indexer_input_detached": True, "cross_layer_reuse": False,
            "draft_block_index_reuse": False,
            "sparse_core_backend": "flash_attn_varlen_gather",
        }
