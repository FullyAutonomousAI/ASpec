# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Anchor sampling shared by the anchored draft training paths."""

import hashlib
from typing import Optional, Tuple

import torch

from aspec.utils.logging import logger


def _candidates(seq_len: int, loss_mask: torch.Tensor, block_size: int, device: torch.device,
                attention_mask: Optional[torch.Tensor] = None):
    """Positions usable as anchors, as (num_candidates, valid, valid_counts, masked_indices).

    An anchor is only usable if its own position and the position right after it are both
    supervised: the block's first prediction target is ``anchor + 1``, so an isolated
    supervised token yields no gradient.
    """
    bsz = loss_mask.shape[0]
    max_anchor = max(seq_len - block_size, 0)
    num_candidates = min(max_anchor + 1, seq_len - 1)

    valid = (loss_mask[:, :num_candidates] > 0.5) & (loss_mask[:, 1 : num_candidates + 1] > 0.5)
    if attention_mask is not None:
        if attention_mask.shape != loss_mask.shape:
            raise ValueError("Anchor attention_mask and loss_mask must have identical shapes")
        # Padded BSHD rows must have the same candidate domain as their native
        # unpadded documents. Physical batch/bucket padding is not context.
        valid = valid & attention_mask[:, :num_candidates].bool() & attention_mask[
            :, block_size - 1:block_size - 1 + num_candidates
        ].bool() & attention_mask[:, block_size:block_size + 1].bool()
    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
    return num_candidates, valid, valid.sum(dim=1), torch.where(valid, indices, seq_len + 1)


def _empty(bsz: int, num_anchors: int, seq_len: int, block_size: int, device: torch.device):
    logger.warning(
        f"Sequence too short for anchor sampling (seq_len={seq_len}, "
        f"block_size={block_size}). Returning dummy anchors so loss is zero."
    )
    return (
        torch.zeros(bsz, num_anchors, dtype=torch.long, device=device),
        torch.zeros(bsz, num_anchors, dtype=torch.bool, device=device),
    )


def anchor_sampling_seed(seed: int, step: int, microbatch: int, replica: int) -> int:
    """Stateless debug RNG key; logical DP replica, never physical CP rank.

    The optimizer step is checkpointed, so exact resume needs no new RNG state.
    Reproducibility assumes the same microbatch assignment; regrouping changes it.
    """
    key = f"{seed}:{step}:{microbatch}:{replica}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2**63)


def validate_injected_anchors(
    anchors: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
    ctx_doc_ids: Optional[torch.Tensor] = None,
    allow_partial_last_target: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate replay inputs before they reach indexing or CP collectives."""
    expected = (loss_mask.shape[0], num_anchors)
    if anchors.shape != expected or keep_mask.shape != expected:
        raise ValueError(f"Injected anchors and keep mask must both have shape {expected}")
    if anchors.dtype not in (torch.int32, torch.int64) or keep_mask.dtype != torch.bool:
        raise ValueError("Injected anchors must be integer and keep mask must be boolean")
    anchors = anchors.to(device=loss_mask.device, dtype=torch.long)
    keep_mask = keep_mask.to(device=loss_mask.device)
    anchors = torch.where(keep_mask, anchors, 0)
    seq_len = loss_mask.shape[1]
    if ((anchors < 0) | (anchors >= max(seq_len - 1, 0)))[keep_mask].any():
        raise ValueError("Kept injected anchors must have an in-bounds first prediction target")
    if not keep_mask.any():
        return anchors, keep_mask
    first_target = (anchors + 1).clamp(max=seq_len - 1)
    valid = (loss_mask.gather(1, anchors) > 0.5) & (
        loss_mask.gather(1, first_target) > 0.5
    )
    if not valid[keep_mask].all():
        raise ValueError("Kept injected anchors and their first targets must be supervised")
    if ctx_doc_ids is not None:
        if ctx_doc_ids.shape != loss_mask.shape:
            raise ValueError("ctx_doc_ids and loss_mask must have identical shapes")
        docs = ctx_doc_ids.to(loss_mask.device)
        last_target = (anchors + block_size - int(allow_partial_last_target)).clamp(max=seq_len - 1)
        anchor_doc = docs.gather(1, anchors)
        same_doc = (anchor_doc >= 0) & (anchor_doc == docs.gather(1, last_target))
        if not same_doc[keep_mask].all():
            raise ValueError("Kept injected anchor blocks must not cross document boundaries")
    return anchors, keep_mask


def document_anchor_slots(document_lengths, num_anchors, block_size, alignment=1):
    """CPU-known structural bounds, padded only for CP anchor ownership."""
    if min(num_anchors, block_size, alignment) < 1:
        raise ValueError("Anchor budget, block size and alignment must be positive")
    slots = tuple(min(num_anchors, max(length - block_size + 1, 0))
                  if length > block_size else 0 for length in document_lengths)
    total = sum(slots)
    padded = max(alignment, ((total + alignment - 1) // alignment) * alignment)
    return slots, total, padded


def sample_document_anchor_positions(
    loss_mask: torch.Tensor,
    ctx_doc_ids: torch.Tensor,
    document_lengths: tuple[int, ...],
    num_anchors: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
    alignment: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the native sampler independently to contiguous packed documents.

    Lengths come from the collator's CPU metadata: no GPU scalar extraction,
    document-by-token matrix, per-document model invocation or rejection loop.
    Storage is O(tokens + documents * anchors). Invalid slots never borrow
    another document's budget. As in the native sampler, the final DSpark
    prediction may be out of bounds and is masked by the objective.
    """
    if (loss_mask.ndim != 2 or loss_mask.shape[0] != 1
            or ctx_doc_ids.shape != loss_mask.shape):
        raise ValueError("Document anchor sampling requires one packed row and aligned document IDs")
    if not document_lengths or any(length <= 0 for length in document_lengths):
        raise ValueError("Packed document lengths must be nonempty and positive")
    seq_len = loss_mask.shape[1]
    if sum(document_lengths) > seq_len:
        raise ValueError("Packed document lengths exceed the row")
    device = loss_mask.device
    n_docs = len(document_lengths)
    slot_counts, total_slots, padded_slots = document_anchor_slots(
        document_lengths, num_anchors, block_size, alignment
    )
    if total_slots == 0:
        return _empty(1, padded_slots, seq_len, block_size, device)
    count, valid, _, _ = _candidates(seq_len, loss_mask, block_size, device)
    lengths = torch.tensor(document_lengths, device=device, dtype=torch.long)
    ends = lengths.cumsum(0)
    docs = ctx_doc_ids[0, :count]
    safe_docs = docs.clamp(min=0, max=n_docs - 1)
    indices = torch.arange(count, device=device)
    valid = (valid[0] & (docs >= 0) & (docs < n_docs)
             & (ctx_doc_ids[0, 1:count + 1] == docs)
             & (lengths[safe_docs] > block_size)
             & (indices <= ends[safe_docs] - block_size))
    counts = torch.zeros(n_docs, device=device, dtype=torch.long)
    counts.scatter_add_(0, safe_docs, valid.long())
    # Documents are contiguous, hence compacting global valid positions also
    # groups them by document without allocating a [documents, tokens] tensor.
    ordered = torch.where(valid, indices, seq_len).sort().values
    starts = counts.cumsum(0) - counts
    divisor = counts.clamp(min=1)
    stride = (counts // num_anchors).clamp(min=1, max=block_size + 1)
    phase = (torch.rand(n_docs, device=device, generator=generator) * divisor).long()
    slot_counts_tensor = torch.tensor(slot_counts, device=device)
    owners = torch.repeat_interleave(
        torch.arange(n_docs, device=device), slot_counts_tensor, output_size=total_slots
    )
    local_slots = torch.arange(total_slots, device=device) - (
        slot_counts_tensor.cumsum(0) - slot_counts_tensor
    )[owners]
    rank = (phase[owners] + local_slots * stride[owners]) % divisor[owners]
    selected = ordered[(starts[owners] + rank).clamp(max=count - 1)]
    keep = local_slots < counts[owners]
    anchors = torch.where(keep, selected, 0)
    pad = padded_slots - total_slots
    return (torch.nn.functional.pad(anchors, (0, pad)).reshape(1, -1),
            torch.nn.functional.pad(keep, (0, pad)).reshape(1, -1))


def sample_anchor_positions(
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
    device: torch.device,
    max_gap: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample anchor positions per sample; returns (anchors, keep_mask).

    Always returns exactly ``num_anchors`` anchor slots so that
    ``Q_LEN = num_anchors * block_size`` is constant across steps, preventing
    FlexAttention recompilation from shape changes. Samples with fewer valid positions
    leave the excess slots masked off.

    ``max_gap`` is how many supervised positions may sit *between* neighbouring anchors,
    so ``max_gap=0`` puts them side by side (5, 6, 7, ...). It matters because anchor ``a``
    supervises token ``a + d`` at unroll depth ``d``: a token is seen at depth ``d`` only if
    the anchor ``d`` positions behind it was also picked. Anchors with a gap of ``g`` leave
    each token ``block_size / (g + 1)`` of its depths, so once ``g`` reaches ``block_size``
    every token is pinned to a single depth.

        max_gap=0     neighbours adjacent; every token in the covered span is supervised
                      at all ``block_size`` depths, as in the dense unroll
        max_gap=k     every (k+1)-th supervised position, over a (k+1)x wider span
        max_gap=None  defaults to ``block_size``: one depth per token

    Gaps count supervised positions, not raw tokens, so slots are never spent on
    unsupervised regions and anchors stay token-adjacent inside a supervised span.

    Args:
        seq_len: sequence length
        loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
        device: torch device
        max_gap: supervised positions permitted between neighbours; None means block_size

    Returns:
        anchors: [B, num_anchors] — sampled anchor positions (sorted)
        keep_mask: [B, num_anchors] — True for valid sampled anchors
    """
    bsz = loss_mask.shape[0]
    if max(seq_len - block_size, 0) == 0:
        return _empty(bsz, num_anchors, seq_len, block_size, device)

    if max_gap is None:
        max_gap = block_size

    num_candidates, _, valid_counts, masked_indices = _candidates(
        seq_len, loss_mask, block_size, device, attention_mask=attention_mask
    )
    # Supervised positions in order; sentinels sort to the back and are masked off below.
    ordered = masked_indices.sort(dim=1).values

    # Spread the budget over the whole sample when it reaches, otherwise pack as tightly
    # as max_gap allows. The window wraps and its phase is drawn over every candidate, so
    # each one is picked with probability num_anchors / valid_count exactly as uniform
    # sampling would. Drawing the phase over the leftover slack instead would pick the
    # middle of a sample far more often than its ends.
    counts = valid_counts.unsqueeze(1).clamp(min=1)
    stride = (valid_counts // num_anchors).clamp(min=1, max=max_gap + 1).unsqueeze(1)
    offset = (torch.rand(bsz, 1, device=device, generator=generator) * counts).long()

    slot = torch.arange(num_anchors, device=device).unsqueeze(0)
    # num_anchors * stride <= valid_count by construction, so the wrap never collides.
    rank = (offset + slot * stride) % counts
    keep_mask = slot < valid_counts.unsqueeze(1)
    anchors = ordered.gather(1, rank.clamp(max=num_candidates - 1))
    return torch.where(keep_mask, anchors, 0), keep_mask
