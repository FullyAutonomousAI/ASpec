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

"""Tiled FP32 softmax L1 loss with recomputation instead of full saved probabilities.

Reimplemented from the public softmax/L1 objective. The existing PyTorch
implementation remains the numerical reference and the non-SM90 path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

_BLOCK = 4096


@triton.jit(do_not_specialize=["TeacherRows"])
def _stats(X, Y, Stats, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr,
           Index=None, TeacherRows=0):
    row, tile = tl.program_id(0), tl.program_id(1)
    # Per-document anchor budgets can make the compact logits exceed 2^31
    # elements. Widen before multiplying by vocabulary, not after overflow.
    row = row.to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    teacher_row = row if Index is None else tl.load(Index + row)
    if Index is not None:
        tl.device_assert((teacher_row >= 0) & (teacher_row < TeacherRows), "L1 teacher row index out of bounds")
    y = tl.load(Y + teacher_row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    mx, my = tl.max(x, 0), tl.max(y, 0)
    sx, sy = tl.sum(tl.exp(x - mx), 0), tl.sum(tl.exp(y - my), 0)
    # A masked vocabulary tile contributes zero mass, not exp(-inf - -inf).
    # A wholly masked row still becomes NaN during the global normalization,
    # matching softmax rather than silently inventing a valid distribution.
    sx = tl.where(mx == -float("inf"), 0.0, sx)
    sy = tl.where(my == -float("inf"), 0.0, sy)
    base = (row * TILES + tile) * 4
    tl.store(Stats + base, mx)
    tl.store(Stats + base + 1, sx)
    tl.store(Stats + base + 2, my)
    tl.store(Stats + base + 3, sy)


@triton.jit
def _merge_stats(Stats, Norms, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    base = (row * TILES + tile) * 4
    mx = tl.load(Stats + base, tile < TILES, other=-float("inf"))
    sx = tl.load(Stats + base + 1, tile < TILES, other=0.0)
    my = tl.load(Stats + base + 2, tile < TILES, other=-float("inf"))
    sy = tl.load(Stats + base + 3, tile < TILES, other=0.0)
    max_x, max_y = tl.max(mx, 0), tl.max(my, 0)
    inv_x = 1.0 / tl.sum(sx * tl.exp(mx - max_x), 0)
    inv_y = 1.0 / tl.sum(sy * tl.exp(my - max_y), 0)
    tl.store(Norms + row * 4, max_x)
    tl.store(Norms + row * 4 + 1, inv_x)
    tl.store(Norms + row * 4 + 2, max_y)
    tl.store(Norms + row * 4 + 3, inv_y)


@triton.jit
def _partials(X, Y, Norms, Parts, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr, Index=None):
    row, tile = tl.program_id(0), tl.program_id(1)
    row = row.to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mx, ix = tl.load(Norms + row * 4), tl.load(Norms + row * 4 + 1)
    my, iy = tl.load(Norms + row * 4 + 2), tl.load(Norms + row * 4 + 3)
    x = tl.load(X + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    teacher_row = row if Index is None else tl.load(Index + row)
    y = tl.load(Y + teacher_row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    p, q = tl.exp(x - mx) * ix, tl.exp(y - my) * iy
    sign = tl.where(p > q, 1.0, tl.where(p < q, -1.0, 0.0))
    offset = (row * TILES + tile) * 4
    tl.store(Parts + offset, tl.sum(tl.abs(p - q), 0))
    tl.store(Parts + offset + 1, tl.sum(tl.where(sign > 0, p, 0.0), 0))
    tl.store(Parts + offset + 2, tl.sum(tl.where(sign < 0, p, 0.0), 0))
    tl.store(Parts + offset + 3, tl.sum(tl.where(sign == 0, p, 0.0), 0))


@triton.jit
def _merge_partials(Parts, Loss, Mean, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    base = (row * TILES + tile) * 4
    loss = tl.load(Parts + base, tile < TILES, other=0.0)
    tl.store(Loss + row, tl.sum(loss, 0))
    for group in tl.static_range(3):
        mass = tl.load(Parts + base + group + 1, tile < TILES, other=0.0)
        tl.store(Mean + row * 3 + group, tl.sum(mass, 0))


@triton.jit
def _backward(X, Y, Norms, Mean, Grad, DX, V: tl.constexpr, BLOCK: tl.constexpr, Index=None):
    row, tile = tl.program_id(0), tl.program_id(1)
    row = row.to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mx, ix = tl.load(Norms + row * 4), tl.load(Norms + row * 4 + 1)
    my, iy = tl.load(Norms + row * 4 + 2), tl.load(Norms + row * 4 + 3)
    x = tl.load(X + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    teacher_row = row if Index is None else tl.load(Index + row)
    y = tl.load(Y + teacher_row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    p, q = tl.exp(x - mx) * ix, tl.exp(y - my) * iy
    sign = tl.where(p > q, 1.0, tl.where(p < q, -1.0, 0.0))
    positive = tl.load(Mean + row * 3)
    negative = tl.load(Mean + row * 3 + 1)
    zero = tl.load(Mean + row * 3 + 2)
    # Equivalent to p*(sign-E_p[sign]), without subtracting two values near
    # one when distributions are sharp. All masses still accumulate in FP32.
    centered = tl.where(sign > 0, 2.0 * negative + zero,
                        tl.where(sign < 0, -2.0 * positive - zero, negative - positive))
    dx = p * centered * tl.load(Grad + row)
    tl.store(DX + row * V + cols, dx, cols < V)


class _TiledL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, indices, replay_cache):
        vocab = logits.shape[-1]
        rows = logits.numel() // vocab
        tiles = triton.cdiv(vocab, _BLOCK)
        if replay_cache:
            norms, mean, loss = replay_cache["statistics"]
            if (loss.shape != logits.shape[:-1] or norms.device != logits.device
                    or replay_cache["vocab"] != vocab):
                raise ValueError("L1 checkpoint statistics belong to a different head chunk")
        else:
            stats = torch.empty((rows, tiles, 4), device=logits.device, dtype=torch.float32)
            norms = torch.empty((rows, 4), device=logits.device, dtype=torch.float32)
            parts = torch.empty((rows, tiles, 4), device=logits.device, dtype=torch.float32)
            loss = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
            mean = torch.empty((rows, 3), device=logits.device, dtype=torch.float32)
            _stats[(rows, tiles)](logits, target, stats, vocab, tiles, _BLOCK, indices,
                                  target.shape[0] if indices is not None else 0,
                                  debug=indices is not None, enable_fp_fusion=False)
            _merge_stats[(rows,)](stats, norms, tiles, triton.next_power_of_2(tiles), enable_fp_fusion=False)
            # Do not fuse p-q asymmetrically: identical distributions must have
            # exactly zero L1 and zero subgradient.
            _partials[(rows, tiles)](logits, target, norms, parts, vocab, tiles, _BLOCK, indices, enable_fp_fusion=False)
            _merge_partials[(rows,)](parts, loss, mean, tiles, triton.next_power_of_2(tiles), enable_fp_fusion=False)
            if replay_cache is not None:
                replay_cache["statistics"] = norms, mean, loss.detach()
                replay_cache["vocab"] = vocab
        ctx.save_for_backward(logits, target, norms, mean, indices)
        # A new view keeps the detached cache free of an output's grad_fn.
        return loss.view_as(loss)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        logits, target, norms, mean, indices = ctx.saved_tensors
        vocab = logits.shape[-1]
        rows = logits.numel() // vocab
        grad_logits = torch.empty_like(logits)
        _backward[(rows, triton.cdiv(vocab, _BLOCK))](
            logits, target, norms, mean, gradient.contiguous(), grad_logits, vocab, _BLOCK, indices,
            enable_fp_fusion=False,
        )
        return grad_logits, None, None, None


def tiled_l1_from_logits(logits: torch.Tensor, target: torch.Tensor, *,
                         target_indices: torch.Tensor | None = None,
                         _replay_cache: dict | None = None) -> torch.Tensor:
    """First-order L1 from compact logits and a frozen teacher.

    Optional flat int64 indices map each Draft row to a row in a 2D teacher
    table, avoiding an expanded vocabulary gather. They do not select or drop
    supervision. Bounds are checked on-device without a host synchronization.

    The internal replay cache belongs to one checkpointed head invocation and
    its deterministic recomputation only. It retains eight FP32 scalars per row,
    never logits/probabilities, and must not be shared across training forwards.
    """
    if not logits.is_cuda or target.device != logits.device:
        raise ValueError("tiled L1 requires CUDA logits on one device")
    if logits.dtype != torch.bfloat16:
        raise ValueError("tiled L1 requires BF16 logits")
    if logits.ndim < 1 or logits.numel() == 0:
        raise ValueError("tiled L1 requires nonempty logits with a vocabulary dimension")
    if target.requires_grad:
        raise ValueError("tiled L1 requires frozen teacher logits")
    if logits.dtype != target.dtype or (
        target_indices is None and logits.shape != target.shape
    ):
        raise ValueError("draft and teacher logits must share shape and dtype")
    if not logits.is_contiguous() or not target.is_contiguous():
        raise ValueError("tiled L1 requires contiguous logits")
    if target_indices is not None:
        if (target.ndim != 2 or target.shape[-1] != logits.shape[-1] or target.shape[0] == 0
                or target_indices.device != logits.device or target_indices.dtype != torch.int64
                or target_indices.ndim != 1 or not target_indices.is_contiguous()
                or target_indices.numel() != logits.numel() // logits.shape[-1]):
            raise ValueError("indexed teacher requires a nonempty 2D table and one contiguous CUDA int64 index per Draft row")
    return _TiledL1.apply(logits, target, target_indices, _replay_cache)
