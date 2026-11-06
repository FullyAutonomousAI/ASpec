# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# See licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.

"""Output-only block selection, adapted from Megatron's DSpark output compaction.

Source: NVIDIA Megatron-LM DSpark output compaction,
``megatron/core/transformer/dspark_prediction.py::_DSparkOutputBlocks``.
ASpec needs neither sequence-parallel padding nor restored token logits: only
additive head terms leave the selection.
Output selection preserves the original head chunks. Dense backbone selection
additionally compacts within each existing CP owner, never across owners.
"""

import torch


def _output_block_counts(keep_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = (keep_mask.shape[1] + chunk_size - 1) // chunk_size
    pad = chunks * chunk_size - keep_mask.shape[1]
    grouped = torch.nn.functional.pad(keep_mask, (0, pad)).reshape(
        keep_mask.shape[0], chunks, chunk_size
    )
    return grouped.sum(-1).amax(0)


def output_block_counts(keep_mask: torch.Tensor, chunk_size: int) -> list[int]:
    """One host transfer for all original head chunks, preserving gradient groups."""
    return _output_block_counts(keep_mask, chunk_size).tolist()


def output_block_indices(keep_mask: torch.Tensor, count: int | None = None) -> torch.Tensor | None:
    """Keep valid blocks in original order; retain masked padding across batch rows.

    One device-to-host scalar establishes the output shape, not one per document.
    Do not infer validity from loss values or resample anchors. For an empty CP
    rank the result has width zero; its backbone still participates in backward.
    """
    if count is None:
        count = int(keep_mask.sum(1).max().item())
    if count == keep_mask.shape[1]:
        return None
    if count == 0:
        return torch.empty((keep_mask.shape[0], 0), dtype=torch.long, device=keep_mask.device)
    return keep_mask.int().argsort(dim=1, descending=True, stable=True)[:, :count]


def select_blocks(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    index = indices.reshape(*indices.shape, *([1] * (value.ndim - 2)))
    return value.gather(1, index.expand(*indices.shape, *value.shape[2:]))


def owner_block_indices(keep_mask: torch.Tensor, owners: int, *, retained: int | None = None) -> torch.Tensor | None:
    """Stable selection inside each contiguous owner, with equal collective widths.

    A single host scalar bounds the shared width across batch rows and owners.
    Keep one masked sentinel even for an entirely empty group: every rank must
    still execute the backbone and its gradient collectives. No owner borrows
    another owner's anchors, and no valid anchor is dropped or resampled.
    """
    if owners < 1 or keep_mask.shape[1] % owners:
        raise ValueError("Query slots must divide evenly across CP owners")
    width = keep_mask.shape[1] // owners
    grouped = keep_mask.reshape(keep_mask.shape[0], owners, width)
    if retained is None:
        retained = max(1, int(grouped.sum(-1).max().item()))
    if not 1 <= retained <= width:
        raise ValueError("Retained owner width must be within the original query slots")
    if retained == width:
        return None
    indices = grouped.int().argsort(dim=-1, descending=True, stable=True)[..., :retained]
    offsets = torch.arange(owners, device=keep_mask.device) * width
    return (indices + offsets[None, :, None]).flatten(1)


def prepare_block_layout(keep_mask: torch.Tensor, owners: int, rank: int, chunk_size: int):
    """One pre-backbone transfer for owner width and this rank's original head chunks.

    Both reductions depend only on sampled validity, not activations. Counts are
    forward-local; no cache survives a new draw, document regrouping or recovery.
    """
    if owners < 1 or not 0 <= rank < owners or keep_mask.shape[1] % owners:
        raise ValueError("Query slots and rank must match contiguous CP owners")
    width = keep_mask.shape[1] // owners
    owner_width = keep_mask.reshape(keep_mask.shape[0], owners, width).sum(-1).amax()
    local_keep = keep_mask[:, rank * width:(rank + 1) * width]
    head_counts = _output_block_counts(local_keep, chunk_size)
    values = torch.cat((owner_width.reshape(1), head_counts)).tolist()
    return owner_block_indices(keep_mask, owners, retained=max(1, values[0])), values[1:]


def restore_blocks(value: torch.Tensor, indices: torch.Tensor, original_width: int) -> torch.Tensor:
    """Restore the original query slots before head chunks and teacher gathers."""
    index = indices.reshape(*indices.shape, *([1] * (value.ndim - 2)))
    return value.new_zeros(value.shape[0], original_width, *value.shape[2:]).scatter(
        1, index.expand_as(value), value
    )


def empty_prediction_terms(hidden: torch.Tensor, heads) -> tuple[torch.Tensor, ...]:
    """Avoid projections but retain reduction hooks for the participating heads."""
    # Multiplication preserves NaN/Inf detection, unlike an empty slice of hidden.
    zero = hidden.float().sum() * 0
    for head in heads:
        if head is not None:
            for parameter in head.parameters():
                if parameter.requires_grad:
                    zero = zero + parameter.reshape(-1)[:0].sum().float()
    positions = hidden.new_zeros(hidden.shape[-2], dtype=torch.float32)
    return zero, zero, zero, positions, positions.clone()
