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

"""Bounded distributed selection helpers for context-sharded sparse attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def max_context_shard_length(metadata, local_length: int, *, alignment: int) -> int:
    """Use one workspace geometry on every owner in a collective loop."""
    if metadata is None or metadata.cp_size == 1:
        return local_length
    if metadata.sequence_length is not None and getattr(metadata, "context_doc_ids", None) is None:
        from aspec.inference.context import context_shard_lengths

        if metadata.layout != "contiguous":
            raise ValueError("compressed sparse attention requires contiguous CP ownership")
        return max(context_shard_lengths(
            metadata.sequence_length, metadata.cp_size, alignment=alignment
        ))

    # Packed rows shard each document independently; their maximum owner
    # length cannot be inferred by sharding the sum of all document lengths.
    # Exchange only its integer capacity once per forward, never activations
    # or per-query state. This also handles metadata without a global length.
    key = ("max_context_shard_length", alignment)
    capacity = metadata._fa4_cache.get(key)
    if capacity is None:
        length = torch.tensor(
            local_length, dtype=torch.long, device=metadata.context_position_ids.device
        )
        dist.all_reduce(length, op=dist.ReduceOp.MAX, group=metadata.group)
        capacity = int(length.item())
        metadata._fa4_cache[key] = capacity
    return capacity


class _AllGatherCandidateScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        candidate_first = scores.movedim(-1, 0).contiguous()
        gathered = torch.empty(
            (candidate_first.shape[0] * world_size, *candidate_first.shape[1:]),
            dtype=scores.dtype,
            device=scores.device,
        )
        dist.all_gather_into_tensor(gathered, candidate_first, group=group)
        ctx.group = group
        ctx.local_candidates = candidate_first.shape[0]
        return gathered.movedim(0, -1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        candidate_first = grad_output.movedim(-1, 0).contiguous()
        local = torch.empty(
            (ctx.local_candidates, *candidate_first.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(local, candidate_first, group=ctx.group)
        return local.movedim(0, -1).contiguous(), None


@dataclass(frozen=True)
class DistributedTopK:
    """Global winners plus this rank's local group index for every winner slot."""

    scores: torch.Tensor
    owners: torch.Tensor
    local_indices: torch.Tensor
    local_winner_indices: torch.Tensor
    valid: torch.Tensor


def _all_gather_candidate_indices(
    indices: torch.Tensor, *, group
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    candidate_first = indices.movedim(-1, 0).contiguous()
    gathered = torch.empty(
        (candidate_first.shape[0] * world_size, *candidate_first.shape[1:]),
        dtype=indices.dtype,
        device=indices.device,
    )
    dist.all_gather_into_tensor(gathered, candidate_first, group=group)
    return gathered.movedim(0, -1).contiguous()


def distributed_topk(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    topk: int,
    group,
) -> DistributedTopK:
    """Select global Top-K from only ``CP * K`` local candidates.

    The function never gathers the context-length score dimension. Candidate
    scores use an autograd-aware gather so the selected loss remains connected
    to the rank that owns each compressed key.
    """

    if scores.ndim != 3 or eligible.shape != scores.shape:
        raise ValueError("distributed sparse scores/eligibility must be [B, Q, local_groups]")
    if topk < 0:
        raise ValueError("distributed sparse topk must be non-negative")
    world_size = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    local_groups = scores.shape[-1]
    local_k = min(topk, local_groups)
    masked = scores.masked_fill(~eligible, -torch.inf)
    if local_k:
        local_scores, local_indices = masked.topk(local_k, dim=-1, sorted=False)
        local_valid = torch.gather(eligible, -1, local_indices)
        local_scores = torch.where(
            local_valid, local_scores, torch.full_like(local_scores, -torch.inf)
        )
        local_indices = torch.where(
            local_valid, local_indices, torch.full_like(local_indices, -1)
        )
    else:
        local_scores = scores[..., :0]
        local_indices = torch.empty(
            *scores.shape[:2], 0, dtype=torch.long, device=scores.device
        )
    if local_k < topk:
        pad = topk - local_k
        local_scores = torch.cat(
            (local_scores, scores.new_full((*scores.shape[:2], pad), -torch.inf)),
            dim=-1,
        )
        local_indices = torch.cat(
            (
                local_indices,
                torch.full(
                    (*scores.shape[:2], pad),
                    -1,
                    dtype=torch.long,
                    device=scores.device,
                ),
            ),
            dim=-1,
        )
    if world_size == 1:
        candidate_scores = local_scores
        candidate_indices = local_indices
    else:
        candidate_scores = _AllGatherCandidateScores.apply(local_scores, group)
        candidate_indices = _all_gather_candidate_indices(local_indices, group=group)
    candidate_owners = torch.arange(
        world_size, device=scores.device, dtype=torch.long
    ).repeat_interleave(topk)
    winner_count = min(topk, candidate_scores.shape[-1])
    if winner_count:
        winner_scores, winner_slots = candidate_scores.topk(
            winner_count, dim=-1, sorted=False
        )
        winner_indices = torch.gather(candidate_indices, -1, winner_slots)
        owners = candidate_owners[winner_slots]
        valid = winner_indices >= 0
        winner_scores = torch.where(
            valid, winner_scores, torch.full_like(winner_scores, -torch.inf)
        )
        owners = torch.where(valid, owners, torch.full_like(owners, -1))
    else:
        shape = (*scores.shape[:2], 0)
        winner_scores = scores.new_empty(shape)
        winner_indices = torch.empty(shape, dtype=torch.long, device=scores.device)
        owners = torch.empty(shape, dtype=torch.long, device=scores.device)
        valid = torch.empty(shape, dtype=torch.bool, device=scores.device)
    local_winners = torch.where(
        valid & owners.eq(rank), winner_indices, torch.full_like(winner_indices, -1)
    )
    return DistributedTopK(
        scores=winner_scores,
        owners=owners,
        local_indices=winner_indices,
        local_winner_indices=local_winners,
        valid=valid,
    )


def distributed_log_softmax(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-softmax over owner-local key columns without gathering them."""

    if scores.shape != eligible.shape:
        raise ValueError("distributed logits and eligibility must have equal shapes")
    world_size = dist.get_world_size(group) if group is not None else 1
    masked = scores.masked_fill(~eligible, -torch.inf)
    if scores.shape[-1]:
        local_max = masked.amax(dim=-1)
    else:
        # A short packed document may leave one CP rank without any complete
        # compression group.  It still participates in the scalar reductions;
        # do not manufacture or gather padded key columns just for that rank.
        local_max = scores.new_full(scores.shape[:-1], -torch.inf)
    global_max = local_max.detach().clone()
    if world_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
    valid = torch.isfinite(global_max)
    safe_max = torch.where(valid, global_max, torch.zeros_like(global_max))
    exponent = torch.where(
        eligible,
        torch.exp(scores - safe_max.unsqueeze(-1)),
        torch.zeros_like(scores),
    )
    denominator = exponent.sum(dim=-1)
    if world_size > 1:
        from torch.distributed.nn.functional import all_reduce

        denominator = all_reduce(denominator, group=group)
    log_denominator = denominator.clamp_min(1e-20).log()
    log_probability = scores - safe_max.unsqueeze(-1) - log_denominator.unsqueeze(-1)
    log_probability = torch.where(
        eligible & valid.unsqueeze(-1),
        log_probability,
        torch.zeros_like(log_probability),
    )
    return log_probability, valid


def owner_local_rows(
    owned_positions: torch.Tensor,
    requested_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global physical rows to one rank's sorted local ownership."""

    if owned_positions.ndim != 2 or requested_positions.ndim < 2:
        raise ValueError("owned and requested positions must include a batch dimension")
    if owned_positions.shape[0] != requested_positions.shape[0]:
        raise ValueError("owned and requested position batches must match")
    if owned_positions.shape[1] == 0:
        return (
            torch.zeros_like(requested_positions, dtype=torch.long),
            torch.zeros_like(requested_positions, dtype=torch.bool),
        )
    requested_flat = requested_positions.flatten(1)
    insertion = torch.searchsorted(
        owned_positions.contiguous(), requested_flat.contiguous(), right=False
    ).view_as(requested_positions)
    safe = insertion.clamp_max(owned_positions.shape[1] - 1)
    found = torch.gather(owned_positions, 1, safe.flatten(1)).view_as(safe)
    return safe, found.eq(requested_positions)


__all__ = [
    "DistributedTopK",
    "distributed_log_softmax",
    "distributed_topk",
    "owner_local_rows",
]
