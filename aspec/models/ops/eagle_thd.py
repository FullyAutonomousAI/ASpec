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

"""Document-isolated EAGLE TTT over owner-sharded, unpadded token streams.

Training mathematics stays in Eagle3Model. This module owns only layout, a
bounded frozen-teacher halo, and FA4/CP execution. No Target feature all-gather.
"""

from functools import lru_cache
from itertools import accumulate
from math import lcm

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aspec.inference.context import context_shard_bounds


class _SequenceHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, inverse):
        ctx.group, ctx.inverse = group, inverse
        size = dist.get_world_size(group)
        batch, tokens, heads, dim = value.shape
        if inverse:
            send = value.reshape(batch, size, tokens // size, heads, dim).movedim(1, 0).contiguous()
        else:
            send = value.reshape(batch, tokens, size, heads // size, dim).movedim(2, 0).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=group)
        if inverse:
            return received.movedim(0, 2).reshape(batch, tokens // size, heads * size, dim)
        return received.movedim(0, 1).reshape(batch, tokens * size, heads // size, dim)

    @staticmethod
    def backward(ctx, gradient):
        return _SequenceHeads.apply(gradient.contiguous(), ctx.group, not ctx.inverse), None, None


def sequence_heads(value, group, *, inverse=False):
    return _SequenceHeads.apply(value, group, inverse)


class EagleTHDLayout:
    """One batch's immutable ownership and integer metadata; never cross-step."""

    def __init__(self, lengths, full_ids, full_loss, local_positions, *, cp_rank=0, cp_size=1, group=None):
        self.lengths = tuple(lengths)
        self.group, self.cp_rank, self.cp_size = group, cp_rank, cp_size
        self.device = full_ids.device
        self.full_ids, self.full_loss = full_ids, full_loss
        self.total = sum(lengths)
        self.max_position = max(lengths)
        self.starts = list(accumulate((0, *lengths[:-1])))
        positions = []
        for rank in range(cp_size):
            pieces = []
            for start, length in zip(self.starts, lengths, strict=True):
                lo, hi = context_shard_bounds(length, rank, cp_size)
                pieces.extend(range(start + lo, start + hi))
            positions.append(pieces)
        self.local_length = max(1, max(map(len, positions)))
        expected = torch.tensor(positions[cp_rank], device=self.device, dtype=torch.long)
        if not torch.equal(local_positions.flatten(), expected):
            raise ValueError("EAGLE feature positions do not match per-document CP ownership")
        self.owned = len(positions[cp_rank])
        self.positions = F.pad(expected, (0, self.local_length - self.owned), value=-1)
        self.global_positions = torch.tensor(
            [p for row in positions for p in row + [-1] * (self.local_length - len(row))],
            device=self.device, dtype=torch.long,
        )
        self.full_docs = torch.repeat_interleave(
            torch.arange(len(lengths), device=self.device),
            torch.tensor(lengths, device=self.device),
        )
        self.full_rope = torch.cat([torch.arange(n, device=self.device) for n in lengths])
        self.docs = self.lookup(self.full_docs, self.positions, invalid=-1)
        self.rope = self.lookup(self.full_rope, self.positions)
        self.global_docs = self.lookup(self.full_docs, self.global_positions, invalid=-1)
        self.global_rope = self.lookup(self.full_rope, self.global_positions)
        self._local_index = torch.full((self.total + 1,), self.local_length, device=self.device, dtype=torch.long)
        self._local_index[expected] = torch.arange(self.owned, device=self.device)
        self._aux = {}
        self._sparse = {}
        self._halo = None
        self._halo_index = None

    def lookup(self, value, positions, *, invalid=0):
        valid = (positions >= 0) & (positions < self.total)
        selected = value.index_select(0, positions.clamp(0, self.total - 1))
        return torch.where(valid, selected, torch.as_tensor(invalid, device=value.device, dtype=value.dtype))

    @classmethod
    def from_batch(cls, batch, *, cp_rank=0, cp_size=1, group=None, teacher_required=True):
        packed = batch.get("ctx_doc_ids") is not None
        if packed:
            docs = batch["ctx_doc_ids"].flatten()
            lengths = torch.unique_consecutive(docs[docs >= 0], return_counts=True)[1].tolist()
            full_ids = batch["input_ids"].flatten()[:sum(lengths)]
            full_loss = batch["loss_mask"].flatten()[:sum(lengths)]
        else:
            lengths = batch["attention_mask"].sum(-1).long().tolist()
            if any(n <= 0 for n in lengths):
                raise ValueError("EAGLE requires nonempty documents")
            full_ids = torch.cat([row[:n] for row, n in zip(batch["input_ids"], lengths, strict=True)])
            full_loss = torch.cat([row[:n] for row, n in zip(batch["loss_mask"].squeeze(-1) if batch["loss_mask"].ndim == 3 else batch["loss_mask"], lengths, strict=True)])
        owned_positions, auxiliary, teacher = [], [], []
        start = 0
        for i, n in enumerate(lengths):
            lo, hi = context_shard_bounds(n, cp_rank, cp_size)
            owned_positions.extend(range(start + lo, start + hi))
            if not packed:
                auxiliary.append(batch["hidden_states"][i, :hi - lo])
                if teacher_required:
                    teacher.append(batch["last_hidden_states"][i, :hi - lo])
            start += n
        local_positions = torch.tensor(owned_positions, device=full_ids.device, dtype=torch.long)
        if packed:
            observed = batch.get("packed_context_positions", local_positions)
            auxiliary = batch["hidden_states"].reshape(-1, batch["hidden_states"].shape[-1])
            teacher = (batch["last_hidden_states"].reshape(-1, batch["last_hidden_states"].shape[-1])
                       if teacher_required else None)
            if not torch.equal(observed.flatten(), local_positions):
                raise ValueError("Packed EAGLE feature ownership mismatch")
        else:
            auxiliary = torch.cat(auxiliary)
            teacher = torch.cat(teacher) if teacher_required else None
        layout = cls(lengths, full_ids, full_loss, local_positions, cp_rank=cp_rank, cp_size=cp_size, group=group)
        if auxiliary.shape[0] != layout.owned or (teacher_required and teacher.shape[0] != layout.owned):
            raise ValueError("EAGLE local features do not match document ownership")
        pad = layout.local_length - layout.owned
        return (layout, F.pad(auxiliary, (0, 0, 0, pad)).unsqueeze(0),
                F.pad(teacher, (0, 0, 0, pad)).unsqueeze(0) if teacher_required else None)

    def prepare_teacher_halo(self, teacher, depth):
        if self.cp_size == 1:
            return
        # Only the first TTT tokens of each owner's document segment can be
        # requested by its predecessor. This is O(CP * docs * TTT), not O(T).
        all_positions = []
        for rank in range(self.cp_size):
            row = []
            for start, n in zip(self.starts, self.lengths, strict=True):
                lo, hi = context_shard_bounds(n, rank, self.cp_size)
                row.extend([start + lo + j if lo + j < hi else -1 for j in range(depth)])
            all_positions.append(row)
        local = torch.tensor(all_positions[self.cp_rank], device=self.device)
        indices = self._local_index[local.clamp_min(0)]
        indices = torch.where(local >= 0, indices, self.local_length)
        extended = F.pad(teacher[0], (0, 0, 0, 1))
        send = extended.index_select(0, indices).contiguous()
        received = [torch.empty_like(send) for _ in range(self.cp_size)]
        dist.all_gather(received, send, group=self.group)
        self._halo = torch.cat(received)
        positions = torch.tensor([p for row in all_positions for p in row], device=self.device)
        self._halo_index = torch.full((self.total + 1,), self._halo.shape[0], device=self.device, dtype=torch.long)
        valid = positions >= 0
        self._halo_index[positions[valid]] = torch.arange(positions.numel(), device=self.device)[valid]
        self._halo = F.pad(self._halo, (0, 0, 0, 1))

    def shifted(self, offset):
        requested = self.positions + offset
        valid = (self.positions >= 0) & (self.lookup(self.full_docs, requested, invalid=-2) == self.docs)
        return requested, valid

    def depth_inputs(self, depth):
        requested, valid = self.shifted(depth + 1)
        ids = self.lookup(self.full_ids, requested)
        # Original TTT shifts loss_mask by depth, while teacher/input shift by
        # depth+1. Cross-document and padded positions are never supervised.
        mask_positions = self.positions + depth
        mask = self.lookup(self.full_loss, mask_positions) * valid
        return torch.where(valid, ids, torch.zeros_like(ids)).unsqueeze(0), mask.unsqueeze(0)

    def teacher_at(self, teacher, depth):
        requested, valid = self.shifted(depth + 1)
        safe = torch.where(valid, requested, self.total)
        indices = self._local_index[safe]
        extended = F.pad(teacher[0], (0, 0, 0, 1))
        result = extended.index_select(0, indices)
        if self.cp_size > 1:
            remote = (indices == self.local_length) & valid
            halo_indices = self._halo_index[safe]
            result = torch.where(remote[:, None], self._halo.index_select(0, halo_indices), result)
        return result.unsqueeze(0)

    def attention_aux(self, depth, head_sharded):
        from aspec.models.ops.fa4_sparse import fixed_mask_aux

        key = depth, head_sharded
        if key not in self._aux:
            query_docs = self.global_docs if self.cp_size > 1 else self.docs
            query_positions = self.global_positions if self.cp_size > 1 else self.positions
            key_docs = self.global_docs if head_sharded else self.docs
            key_positions = self.global_positions if head_sharded else self.positions
            width = key_positions.numel()
            self._aux[key] = [
                fixed_mask_aux(x.unsqueeze(0)) for x in (
                    query_docs, query_positions, key_docs.repeat(depth + 1),
                    key_positions.repeat(depth + 1),
                    torch.arange(depth + 1, device=self.device).repeat_interleave(width),
                )
            ]
        return self._aux[key]


@lru_cache(maxsize=128)
def eagle_fixed_mask(window=None):
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def read(aux, batch, index: "cute.TensorSSA", length):
        fragment = cute.make_rmem_tensor(1, cutlass.Int32)
        fragment.store(index)
        value = cute.make_rmem_tensor(1, cutlass.Int32)
        value[0] = aux[batch[0], cutlass.min(fragment[0], length - 1)]
        return value.load()

    @cute.jit
    def mask(batch, head, q, k, seqlen_info, aux_tensors):
        del head
        qdocs, qpos, kdocs, kpos, kdepth = aux_tensors
        qdoc = read(qdocs, batch, q, seqlen_info.seqlen_q)
        qposition = read(qpos, batch, q, seqlen_info.seqlen_q)
        kdoc = read(kdocs, batch, k, seqlen_info.seqlen_k)
        kposition = read(kpos, batch, k, seqlen_info.seqlen_k)
        depth = read(kdepth, batch, k, seqlen_info.seqlen_k)
        context = (depth == 0) & (kposition <= qposition)
        if cutlass.const_expr(window is not None):
            context = context & (kposition > qposition - window)
        return (qdoc >= 0) & (qdoc == kdoc) & (context | (kposition == qposition))
    return mask


def exchange_eagle_kv(key, value, layout):
    target_kv = lcm(key.shape[2], layout.cp_size)
    factor = target_kv // key.shape[2]
    key = key.repeat_interleave(factor, dim=2) if factor > 1 else key
    value = value.repeat_interleave(factor, dim=2) if factor > 1 else value
    return sequence_heads(key, layout.group), sequence_heads(value, layout.group)


def eagle_fa4_attention(query, key, value, layout, depth, observation, *, kv_exchanged=False,
                        softmax_scale=None, window=None):
    """FA4 fixed sparse, with head/sequence exchange when heads permit it."""
    from flash_attn.cute.interface import flash_attn_func

    from aspec.models.ops.direct_cp import (
        all_gather_query_rows,
        all_to_all_query_partials,
        merge_attention_partials,
    )
    from aspec.models.ops.fa4_sparse import build_fixed_sparse_pair

    head_sharded = layout.cp_size > 1 and query.shape[2] % layout.cp_size == 0
    if layout.cp_size > 1:
        if head_sharded:
            # GQA heads need only the minimum replication needed by the group.
            query = sequence_heads(query, layout.group)
            if not kv_exchanged:
                key, value = exchange_eagle_kv(key, value, layout)
        else:
            query = all_gather_query_rows(query, group=layout.group)
    aux = layout.attention_aux(depth, head_sharded)
    sparse_key = depth, head_sharded, query.shape[-1], value.shape[-1], window
    sparse = layout._sparse.get(sparse_key)
    if sparse is None:
        sparse = build_fixed_sparse_pair(
            mask_mod=eagle_fixed_mask(window), aux_tensors=aux, batch_size=1,
            query_length=query.shape[1], key_length=key.shape[1],
            device=query.device, query_head_dim=query.shape[-1], value_head_dim=value.shape[-1],
        )
        layout._sparse[sparse_key] = sparse
    output, lse = flash_attn_func(
        query, key, value, causal=False, mask_mod=eagle_fixed_mask(window), aux_tensors=aux,
        softmax_scale=softmax_scale,
        block_sparse_tensors=sparse.forward, block_sparse_tensors_bwd=sparse.backward,
        return_lse=True,
    )[:2]
    if layout.cp_size > 1:
        if head_sharded:
            output = sequence_heads(output, layout.group, inverse=True)
        else:
            partial = all_to_all_query_partials(output.transpose(1, 2), group=layout.group)
            lse_parts = all_to_all_query_partials(lse, group=layout.group)
            output = merge_attention_partials(partial, lse_parts)[0].transpose(1, 2).to(query.dtype)
    observation.update(
        qkv_format="thd", fa4_mode="fixed", backward_strategy="fa4_fixed_sparse",
        backward_observed=False, empty_local_partition=layout.owned == 0,
        context_parallel_size=layout.cp_size, context_parallel_rank=layout.cp_rank,
        cp_strategy="head_sequence_all_to_all" if head_sharded else "query_partials" if layout.cp_size > 1 else "local",
        query_owned_tokens=layout.owned, query_collective_tokens=query.shape[1],
        query_collective_heads=query.shape[2],
        collective_padding_tokens=layout.local_length * layout.cp_size - layout.total,
        teacher_halo_rows=0 if layout._halo is None else layout._halo.shape[0] - 1,
        target_feature_all_gather=False,
    )
    if output.requires_grad:
        output.register_hook(lambda grad: observation.update(backward_observed=True) or grad)
    output = torch.where((layout.positions >= 0)[None, :, None, None], output, 0.0)
    return output
