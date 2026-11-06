# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted by ASpec Authors from NVIDIA cuDNN Frontend cb8cec8.

"""ASpec-owned CuTeDSL adjoint for the DSpark Q/K preparation kernel.

The main launch computes the exact RoPE adjoint, RMSNorm input gradients, and
per-CTA FP32 norm-weight partials. A second launch reduces those partials in a
fixed order, so repeated runs are deterministic without atomics.
"""

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as nvvm

from aspec.kernels.cutedsl._primitives import (
    abs_max_tree,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp16,
    lane_group_sum,
    launch_dependent_grids,
    ld_global,
    ld_global_v4,
    ld_shared_v4,
    opaque_f32_zero,
    st_global,
    st_global_v4,
    st_shared_v4,
    wait_on_dependent_grids,
    warp_abs_max_f32_shfl,
)
from aspec.kernels.cutedsl.dspark_qk_norm_rope import (
    ACCESS_BYTES,
    ELEMS_PER_ACCESS,
    _fake,
    lanes_per_row,
    validate_shape,
    vec_chunks,
)

DEFAULT_THREADS_PER_CTA = 128
# The norm arm is register-pressure bound and uses one row per lane group; the
# RoPE-only arm has a smaller live set and benefits from two rows in flight.
DEFAULT_ROWS_PER_GROUP = 1
DEFAULT_ROWS_PER_GROUP_ROPE_ONLY = 2
DEFAULT_CONST_HEAD_COUNTS = True
# Cap the persistent main grid at the measured eight resident CTAs per SM. The
# runtime derives the SM count from the selected device; explicit caller policy
# remains available for architecture-specific validation.
CTAS_PER_SM = 8
DEFAULT_N_CTAS_POLICY = "sm_fill"
_BPE = 2
# The deterministic weight-gradient reduction maps eight adjacent FP32 columns
# across 128 row lanes. This preserves the summation order while coalescing the
# partial loads. Numerical comparisons and supported test cases are maintained
# in tests/test_dspark_qk_cutedsl.py.
REDUCE_COLS = 8
REDUCE_LANES = 128
REDUCE_UNROLL = 0

_FAKE_STREAM = None


def _check_one_cuda_device(name: str, tensor: torch.Tensor, others) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
    for other_name, other in others:
        if other is not None and other.device != tensor.device:
            raise ValueError(
                f"{other_name} must be on {tensor.device} with {name}, got {other.device}"
            )


def _multiprocessor_count(device: int) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)




@cute.jit
def _row_class(
    dy_base: cutlass.Int64,
    dy_tok: cutlass.Int64,  # token strides in ELEMENTS (head stride is D, elem stride 1: baked)
    x_base: cutlass.Int64,
    x_tok: cutlass.Int64,
    x2_base: cutlass.Int64,
    x2_tok: cutlass.Int64,
    rstd_base: cutlass.Int64,
    w_base: cutlass.Int64,
    cos_base: cutlass.Int64,
    cos_tok: cutlass.Int64,
    sin_base: cutlass.Int64,
    sin_tok: cutlass.Int64,
    pos_base: cutlass.Int64,
    pos2_base: cutlass.Int64,
    out_base: cutlass.Int64,
    out_tok: cutlass.Int64,
    out2_base: cutlass.Int64,
    out2_tok: cutlass.Int64,
    dw_base: cutlass.Int64,  # THIS CTA's [D] partial row (already offset)
    sl_base: cutlass.Int64,
    n_class_rows: cutlass.Int32,
    n_iters: cutlass.Int32,
    cta_in_class: cutlass.Int32,
    n_ctas_class: cutlass.Int32,
    h: cutlass.Int32,
    s: cutlass.Int32,
    context_len: cutlass.Int32,
    draft_len: cutlass.Int32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    apply_norm: cutlass.Constexpr[bool],
    want_dw: cutlass.Constexpr[bool],
    has_seq_lens: cutlass.Constexpr[bool],
    indexed_rope: cutlass.Constexpr[bool],
    split_tokens: cutlass.Constexpr[bool],
    batch_one: cutlass.Constexpr[bool],
    is_copy: cutlass.Constexpr[bool],
    sdw,  # the ONE sDwPartial SMEM Array of the kernel (None without want_dw)
    mAmaxOut: Optional[cute.Tensor],  # [>= grid] fp32: partials[cta] = max |stored words| of this CTA (plain store at CTA end); None = no fold
    cta: cutlass.Int32,  # the GRID-level block index: the partial this CTA owns
    sred,  # the ONE sRed SMEM Array of the kernel, fp32 [warps] (None without want_amax)
) -> None:
    """The persistent loop of ONE row class (Q, K or V) for one CTA.

    Takes base addresses and element strides as scalars (the tensors' facts,
    read off ``.iterator.toint()`` / ``.stride`` in the kernel) so the three
    class arms specialise on ``h_ct`` / ``apply_norm`` / ``is_copy`` with one
    body. Row ``row`` of the class is token ``row // h``, head ``row % h``;
    iteration ``it`` of CTA ``cta_in_class`` covers row group
    ``cta_in_class + it * n_ctas_class`` (``rows_per_cta`` rows). ``sdw`` is
    the kernel's single ``sDwPartial`` allocation (SMEM table): allocating it
    here would instantiate it once per inlined arm.
    """
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    n_acc = cutlass.const_expr(chunks * ELEMS_PER_ACCESS)

    want_amax = cutlass.const_expr(mAmaxOut is not None)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    warp_lane = tidx % cutlass.Int32(32)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    bpe = cutlass.Int64(_BPE)
    d64 = cutlass.Int64(d)
    in_rope = lane < cutlass.Int32(rope_lanes) if cutlass.const_expr(rope_dim > 0) else cutlass.Boolean(False)
    zero = cutlass.Float32(0.0)
    inv_d = cutlass.Float32(1.0 / d)

    # ONE [D] dW accumulator per lane (8 fp32 per chunk), live across the whole
    # persistent loop: the class is CTA-uniform, so it never mixes Q and K.
    dw_acc = [cutlass.Float32(0.0) for _ in range(n_acc)]
    # The amax fold of the STORED words, per lane across the whole loop (want_amax only; module docstring).
    amax_lane = opaque_f32_zero()

    for it in cutlass.range(n_iters):
        g_idx = cta_in_class + it * n_ctas_class
        row0 = (g_idx * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)

        # --- PASS 1: every load this thread makes for its rows -----------------
        rows = []
        deads = []
        tokens = []
        out_addrs = []
        dys = []
        xs = []
        rstds = []
        for r in cutlass.range_constexpr(rows_per_group):
            row = row0 + cutlass.Int32(r)
            row_r = row if row < n_class_rows else n_class_rows - cutlass.Int32(1)
            token = row_r // _h
            head = row_r % _h
            tok64 = token.to(cutlass.Int64)
            head64 = head.to(cutlass.Int64)
            source_token = token
            source_x_base = x_base
            source_x_tok = x_tok
            source_out_base = out_base
            source_out_tok = out_tok
            source_pos_base = pos_base
            if cutlass.const_expr(split_tokens):
                total_len = context_len + draft_len
                if cutlass.const_expr(batch_one):
                    bidx = cutlass.Int32(0)
                    pos = token
                else:
                    bidx = token // total_len
                    pos = token - bidx * total_len
                if pos < context_len:
                    source_token = bidx * context_len + pos
                else:
                    source_token = bidx * draft_len + (pos - context_len)
                    source_x_base = x2_base
                    source_x_tok = x2_tok
                    source_out_base = out2_base
                    source_out_tok = out2_tok
                    source_pos_base = pos2_base
            source_tok64 = source_token.to(cutlass.Int64)
            dead = cutlass.Boolean(False)
            if cutlass.const_expr(has_seq_lens):
                bidx = token // s
                pos = token - bidx * s
                slen = ld_global(sl_base + bidx.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
                dead = pos >= slen
            dy_addr = dy_base + (tok64 * dy_tok + head64 * d64) * bpe
            out_addr = source_out_base + (source_tok64 * source_out_tok + head64 * d64) * bpe
            row_dy = []
            row_x = []
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                words = ld_global_v4(dy_addr + off, cutlass.Int32)
                if cutlass.const_expr(is_copy):
                    row_dy.append(list(words))  # raw 16-B words: the copy never unpacks
                else:
                    row_dy.append([v for w in words for v in f16x2_to_f32(w, dtype=io_dtype)])
            rstd = cutlass.Float32(1.0)
            if cutlass.const_expr(apply_norm):
                x_addr = source_x_base + (source_tok64 * source_x_tok + head64 * d64) * bpe
                for c in cutlass.range_constexpr(chunks):
                    off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                    row_x.append([v for w in ld_global_v4(x_addr + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
                # scalar 4-B load: a v2/v4 needs an R*4-B aligned K-side offset (see the docstring)
                rstd = ld_global(rstd_base + row_r.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Float32)
            rows.append(row)
            deads.append(dead)
            rope_token = token
            if cutlass.const_expr(indexed_rope):
                rope_token = ld_global(source_pos_base + source_tok64 * cutlass.Int64(8), cutlass.Int64).to(cutlass.Int32)
            tokens.append(rope_token)
            out_addrs.append(out_addr)
            dys.append(row_dy)
            xs.append(row_x)
            rstds.append(rstd)

        # --- PASS 2: adjoint RoPE, RMSNorm backward, dW fold, store ----------------
        for r in cutlass.range_constexpr(rows_per_group):
            valid = rows[r] < n_class_rows
            if cutlass.const_expr(is_copy):
                # V class: dx = dy, bit-exact. The dead-row select is on the raw words
                # (0x00000000 == two +0.0 halves in either io dtype).
                stored = []
                for c in cutlass.range_constexpr(chunks):
                    words = list(dys[r][c])
                    if cutlass.const_expr(has_seq_lens):
                        words = [(cutlass.Int32(0) if deads[r] else w) for w in words]
                    stored.append(words)
                    if valid:
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        st_global_v4(out_addrs[r] + off, words, cutlass.Int32)
                if cutlass.const_expr(want_amax):
                    # max |dV| over the STORED words (what the dqkvg quantize reads next), valid rows only
                    row_amax = abs_max_tree(
                        [v for c in range(chunks) for w in range(ELEMS_PER_ACCESS // 2) for v in f16x2_to_f32(stored[c][w], dtype=io_dtype)]
                    )
                    amax_lane = fmax_f32(amax_lane, row_amax) if valid else amax_lane
            else:
                g = [list(dys[r][c]) for c in range(chunks)]
                if cutlass.const_expr(rope_dim > 0):
                    # cos/sin at their point of use, on the rope lanes only (the
                    # forward's deferred-load lesson: both are L2 hits by construction).
                    cos_v = [zero] * ELEMS_PER_ACCESS
                    sin_v = [zero] * ELEMS_PER_ACCESS
                    if in_rope:
                        tok64 = tokens[r].to(cutlass.Int64)
                        cb = cos_base + tok64 * cos_tok * bpe + lane_off
                        sb = sin_base + tok64 * sin_tok * bpe + lane_off
                        cos_v = [v for w in ld_global_v4(cb, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)]
                        sin_v = [v for w in ld_global_v4(sb, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)]
                    # Exact adjoint: dx = dy*cos - rotate_half(dy*sin). The shuffle is
                    # unconditional (every lane of the warp must reach shfl.sync; the
                    # non-rope lanes carry sin = 0) and only the rope lanes keep the result.
                    rotated = []
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        t_i = g[0][i] * sin_v[i]
                        partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, t_i, cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                        signed = partner if lane < cutlass.Int32(half_lanes) else -partner  # the forward's sign, FLIPPED
                        rotated.append(cute.math.fma(g[0][i], cos_v[i], signed))
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        g[0][i] = rotated[i] if in_rope else g[0][i]

                dx = []
                if cutlass.const_expr(apply_norm):
                    # weights at their point of use ([D], shared by every row of the class)
                    ws = []
                    for c in cutlass.range_constexpr(chunks):
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        ws.append([v for w in ld_global_v4(w_base + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
                    rstd = rstds[r]
                    x_hat = [[xs[r][c][i] * rstd for i in range(ELEMS_PER_ACCESS)] for c in range(chunks)]
                    gw = [[g[c][i] * ws[c][i] for i in range(ELEMS_PER_ACCESS)] for c in range(chunks)]
                    acc = zero
                    for c in cutlass.range_constexpr(chunks):
                        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                            acc = cute.math.fma(gw[c][i], x_hat[c][i], acc)
                    m = lane_group_sum(acc, lanes) * inv_d
                    for c in cutlass.range_constexpr(chunks):
                        dx.append([rstd * cute.math.fma(-x_hat[c][i], m, gw[c][i]) for i in range(ELEMS_PER_ACCESS)])
                    if cutlass.const_expr(want_dw):
                        # dead / tail rows contribute EXACTLY 0: select BEFORE the add (a
                        # NaN dy on a dead row must not poison dW), never `* 0`.
                        keep = valid & (~deads[r]) if cutlass.const_expr(has_seq_lens) else valid
                        for c in cutlass.range_constexpr(chunks):
                            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                                dw_acc[c * ELEMS_PER_ACCESS + i] = (
                                    cute.math.fma(
                                        g[c][i],
                                        x_hat[c][i],
                                        dw_acc[c * ELEMS_PER_ACCESS + i],
                                    )
                                    if keep
                                    else dw_acc[c * ELEMS_PER_ACCESS + i]
                                )
                else:
                    # RoPE-only: dx IS g; [rope_dim, D) goes back through fp32_to_fp16
                    # untouched by any fp32 op -> a bit-exact copy (asserted by the tests).
                    dx = g
                if cutlass.const_expr(has_seq_lens):
                    for c in cutlass.range_constexpr(chunks):
                        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                            dx[c][i] = zero if deads[r] else dx[c][i]
                # ONE rounding at the store: the packed io-dtype words (also what the amax fold reads back)
                packed = [[fp32_to_fp16(dx[c][2 * i], dx[c][2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)] for c in range(chunks)]
                if valid:
                    for c in cutlass.range_constexpr(chunks):
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        st_global_v4(out_addrs[r] + off, packed[c], cutlass.Int32)
                if cutlass.const_expr(want_amax):
                    # max |dx| over the STORED words (unpacked), valid rows only -- a dead row's dx is the selected zero
                    row_amax = abs_max_tree(
                        [v for c in range(chunks) for w in range(ELEMS_PER_ACCESS // 2) for v in f16x2_to_f32(packed[c][w], dtype=io_dtype)]
                    )
                    amax_lane = fmax_f32(amax_lane, row_amax) if valid else amax_lane

    # --- CTA end: the amax fold -- every lane of the warp reaches the butterfly (the loop bound is CTA-uniform; the
    # row loop diverges only around stores); the warps combine through sRed behind ONE barrier (barrier table) and
    # thread 0 STORES this CTA's partial -- a plain store, no atomic (module docstring) ----
    if cutlass.const_expr(want_amax):
        warps = cutlass.const_expr(threads_per_cta // 32)
        warp_id = tidx // cutlass.Int32(32)
        warp_amax = warp_abs_max_f32_shfl(amax_lane)
        if warp_lane == cutlass.Int32(0):
            sred.subview(warp_id).store(warp_amax)
        nvvm.barrier_cta_sync()
        if tidx == cutlass.Int32(0):
            m = sred.subview(0).load()
            for w in cutlass.range_constexpr(1, warps):
                m = fmax_f32(m, sred.subview(w).load())
            st_global(mAmaxOut.iterator.toint() + cta.to(cutlass.Int64) * cutlass.Int64(4), m, cutlass.Float32)

    # --- CTA end: combine the four lane groups' dW in FIXED order, store the partial ----
    if cutlass.const_expr(want_dw):
        # sDwPartial -- see the module docstring's SMEM table: interleaved so 8
        # consecutive lanes of a v4 phase cover one 128-B line (4 wavefronts / 512 B).
        # Allocated once by the kernel and passed in (one instance, not one per arm).
        sDwPartial = sdw
        quads = cutlass.const_expr(ELEMS_PER_ACCESS // 4)
        if grp > cutlass.Int32(0):
            for c in cutlass.range_constexpr(chunks):
                for j in cutlass.range_constexpr(quads):
                    p = sDwPartial.data_ptr(grp * cutlass.Int32(d) + cutlass.Int32(c * lanes * ELEMS_PER_ACCESS + j * lanes * 4) + lane * cutlass.Int32(4))
                    st_shared_v4(p, dw_acc[c * ELEMS_PER_ACCESS + 4 * j : c * ELEMS_PER_ACCESS + 4 * j + 4], cutlass.Float32)
        # ONE named barrier per CTA (barrier table row): all threads_per_cta threads
        # arrive -- the persistent loop bound is CTA-uniform. Generic proxy both sides.
        nvvm.barrier_cta_sync()
        if grp == cutlass.Int32(0):
            tot = list(dw_acc)
            for gi in cutlass.range_constexpr(1, groups_per_cta):
                for c in cutlass.range_constexpr(chunks):
                    for j in cutlass.range_constexpr(quads):
                        p = sDwPartial.data_ptr(cutlass.Int32(gi * d + c * lanes * ELEMS_PER_ACCESS + j * lanes * 4) + lane * cutlass.Int32(4))
                        v = ld_shared_v4(p, cutlass.Float32)
                        for q in cutlass.range_constexpr(4):
                            tot[c * ELEMS_PER_ACCESS + 4 * j + q] = tot[c * ELEMS_PER_ACCESS + 4 * j + q] + v[q]
            for c in cutlass.range_constexpr(chunks):
                for j in cutlass.range_constexpr(quads):
                    elem = cutlass.Int64(c * lanes * ELEMS_PER_ACCESS + 4 * j) + lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                    st_global_v4(dw_base + elem * cutlass.Int64(4), tot[c * ELEMS_PER_ACCESS + 4 * j : c * ELEMS_PER_ACCESS + 4 * j + 4], cutlass.Float32)


@cute.kernel
def frost_qk_norm_rope_bwd(
    mDQ: cute.Tensor,  # [T, H_q,  D] compact: B4's dQ
    mDK: cute.Tensor,  # [T, H_kv, D] compact: B4's dK
    mDV: cute.Tensor,  # [T, H_kv, D] compact: B4's dV
    mXq: Optional[cute.Tensor],  # q_pre: proj_slab band (token stride N) or compact; None under apply_norm=False
    mXk: Optional[cute.Tensor],  # k_pre
    mXk2: Optional[cute.Tensor],  # optional second K source (DSpark draft K)
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32 (saved); None under apply_norm=False
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32
    mWq: Optional[cute.Tensor],  # [D] norm weight; None under apply_norm=False (presence switch)
    mWk: Optional[cute.Tensor],  # [D]
    mCos: cute.Tensor,  # [T, rope_dim]
    mSin: cute.Tensor,  # [T, rope_dim]
    mPosQ: Optional[cute.Tensor],  # optional int64 Q positions
    mPosK: Optional[cute.Tensor],  # optional int64 first-K-source positions
    mPosK2: Optional[cute.Tensor],  # optional int64 second-K-source positions
    mOutQ: cute.Tensor,  # the dqkvg Q band (token stride N, head stride D); MAY alias mDQ
    mOutK: cute.Tensor,  # the dqkvg K band; MAY alias mDK
    mOutK2: Optional[cute.Tensor],  # optional second K-gradient destination
    mOutV: cute.Tensor,  # the dqkvg V band
    mDWq: Optional[cute.Tensor],  # [n_ctas_q, D] fp32 per-CTA partials (None: no dW)
    mDWk: Optional[cute.Tensor],  # [n_ctas_k, D]
    mSeqLens: Optional[cute.Tensor],  # [B] int32, or None (dense: the select is folded out)
    mAmaxOut: Optional[cute.Tensor],  # [>= n_ctas_q + n_ctas_k + n_ctas_v] fp32 OUT: per-CTA partials of max |stored dqkvg words|; None = no fold
    n_ctas_q: cutlass.Int32,
    n_ctas_k: cutlass.Int32,
    n_ctas_v: cutlass.Int32,
    n_q_rows: cutlass.Int32,
    n_k_rows: cutlass.Int32,  # == the V row count (both H_kv)
    s: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    context_len: cutlass.Int32,
    draft_len: cutlass.Int32,
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    batch_one: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Three CTA classes over one grid (Q | K | V), each a persistent row loop; see the module docstring."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    apply_norm = cutlass.const_expr(mWq is not None)
    split_k = cutlass.const_expr(mXk2 is not None)
    indexed_rope = cutlass.const_expr(mPosQ is not None)
    want_dw = cutlass.const_expr(mDWq is not None)
    has_seq_lens = cutlass.const_expr(mSeqLens is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    rows_per_iter = cutlass.const_expr((threads_per_cta // lanes) * rows_per_group)
    io_dtype = mDQ.element_type
    i64_0 = cutlass.Int64(0)

    cta = cutlass.Int32(cute.arch.block_idx()[0])
    sl_base = mSeqLens.iterator.toint() if cutlass.const_expr(has_seq_lens) else i64_0
    cos_base = mCos.iterator.toint()
    cos_tok = cutlass.Int64(mCos.stride[0])
    sin_base = mSin.iterator.toint()
    sin_tok = cutlass.Int64(mSin.stride[0])
    d_bytes = cutlass.Int64(d * 4)
    rpi = cutlass.Int32(rows_per_iter)
    # The ONE sDwPartial (SMEM table: fp32 groups_per_cta x D, 4 KiB at D=256 /
    # 128 threads). Allocated here, not in _row_class: that body is inlined per
    # class arm and an allocation inside it ships once per arm (8 KiB).
    sdw = cutlass.Array(cutlass.Float32, (threads_per_cta // lanes) * d, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_dw) else None
    # The ONE sRed of the amax fold (SMEM table): the warp maxima, combined by thread 0 into this CTA's partial.
    want_amax = cutlass.const_expr(mAmaxOut is not None)
    sred = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_amax) else None

    if cta < n_ctas_q:
        # ---- Q class -------------------------------------------------------------
        n_groups = (n_q_rows + rpi - cutlass.Int32(1)) // rpi
        n_iters = (n_groups - cta + n_ctas_q - cutlass.Int32(1)) // n_ctas_q
        x_base = mXq.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        x_tok = cutlass.Int64(mXq.stride[0]) if cutlass.const_expr(apply_norm) else i64_0
        rstd_base = mRstdQ.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        w_base = mWq.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        dw_base = (mDWq.iterator.toint() + cta.to(cutlass.Int64) * d_bytes) if cutlass.const_expr(want_dw) else i64_0
        _row_class(
            mDQ.iterator.toint(),
            cutlass.Int64(mDQ.stride[0]),
            x_base,
            x_tok,
            i64_0,
            i64_0,
            rstd_base,
            w_base,
            cos_base,
            cos_tok,
            sin_base,
            sin_tok,
            mPosQ.iterator.toint() if cutlass.const_expr(indexed_rope) else i64_0,
            i64_0,
            mOutQ.iterator.toint(),
            cutlass.Int64(mOutQ.stride[0]),
            i64_0,
            i64_0,
            dw_base,
            sl_base,
            n_q_rows,
            n_iters,
            cta,
            n_ctas_q,
            h_q,
            s,
            context_len,
            draft_len,
            io_dtype,
            h_q_ct,
            const_head_counts,
            d,
            rope_dim,
            threads_per_cta,
            rows_per_group,
            apply_norm,
            want_dw,
            has_seq_lens,
            indexed_rope,
            False,
            batch_one,
            False,
            sdw,
            mAmaxOut,
            cta,
            sred,
        )
    else:
        if cta < n_ctas_q + n_ctas_k:
            # ---- K class ---------------------------------------------------------
            cta_k = cta - n_ctas_q
            n_groups = (n_k_rows + rpi - cutlass.Int32(1)) // rpi
            n_iters = (n_groups - cta_k + n_ctas_k - cutlass.Int32(1)) // n_ctas_k
            x_base = mXk.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            x_tok = cutlass.Int64(mXk.stride[0]) if cutlass.const_expr(apply_norm) else i64_0
            x2_base = mXk2.iterator.toint() if cutlass.const_expr(apply_norm and split_k) else i64_0
            x2_tok = cutlass.Int64(mXk2.stride[0]) if cutlass.const_expr(apply_norm and split_k) else i64_0
            rstd_base = mRstdK.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            w_base = mWk.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            dw_base = (mDWk.iterator.toint() + cta_k.to(cutlass.Int64) * d_bytes) if cutlass.const_expr(want_dw) else i64_0
            _row_class(
                mDK.iterator.toint(),
                cutlass.Int64(mDK.stride[0]),
                x_base,
                x_tok,
                x2_base,
                x2_tok,
                rstd_base,
                w_base,
                cos_base,
                cos_tok,
                sin_base,
                sin_tok,
                mPosK.iterator.toint() if cutlass.const_expr(indexed_rope) else i64_0,
                mPosK2.iterator.toint() if cutlass.const_expr(indexed_rope and split_k) else i64_0,
                mOutK.iterator.toint(),
                cutlass.Int64(mOutK.stride[0]),
                mOutK2.iterator.toint() if cutlass.const_expr(split_k) else i64_0,
                cutlass.Int64(mOutK2.stride[0]) if cutlass.const_expr(split_k) else i64_0,
                dw_base,
                sl_base,
                n_k_rows,
                n_iters,
                cta_k,
                n_ctas_k,
                h_kv,
                s,
                context_len,
                draft_len,
                io_dtype,
                h_kv_ct,
                const_head_counts,
                d,
                rope_dim,
                threads_per_cta,
                rows_per_group,
                apply_norm,
                want_dw,
                has_seq_lens,
                indexed_rope,
                split_k,
                batch_one,
                False,
                sdw,
                mAmaxOut,
                cta,
                sred,
            )
        else:
            # ---- V class: the band copy (no norm, no RoPE, no dW) --------------------
            cta_v = cta - n_ctas_q - n_ctas_k
            n_groups = (n_k_rows + rpi - cutlass.Int32(1)) // rpi
            n_iters = (n_groups - cta_v + n_ctas_v - cutlass.Int32(1)) // n_ctas_v
            _row_class(
                mDV.iterator.toint(),
                cutlass.Int64(mDV.stride[0]),
                i64_0,
                i64_0,
                i64_0,
                i64_0,
                i64_0,
                i64_0,
                cos_base,
                cos_tok,
                sin_base,
                sin_tok,
                i64_0,
                i64_0,
                mOutV.iterator.toint(),
                cutlass.Int64(mOutV.stride[0]),
                i64_0,
                i64_0,
                i64_0,
                sl_base,
                n_k_rows,
                n_iters,
                cta_v,
                n_ctas_v,
                h_kv,
                s,
                context_len,
                draft_len,
                io_dtype,
                h_kv_ct,
                const_head_counts,
                d,
                0,
                threads_per_cta,
                rows_per_group,
                False,
                False,
                has_seq_lens,
                False,
                False,
                batch_one,
                True,
                sdw,
                mAmaxOut,
                cta,
                sred,
            )

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def dw_reduce_column(
    p_base: cutlass.Int64,  # this class's [n, D] fp32 partials plane
    o_base: cutlass.Int64,  # this class's [D] fp32 output
    n: cutlass.Int32,  # the plane's rows
    col: cutlass.Int32,  # the column this thread sums (>= d: an idle thread that still takes the barrier)
    lane: cutlass.Int32,  # the row-lane (residue class) of this thread, in [0, lanes)
    j: cutlass.Int32,  # the column slot within the block, in [0, cols)
    d: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    sPart,  # fp32 SMEM Array, lanes x cols
) -> None:
    """The fixed-order reduce of ONE column, by the ``lanes`` threads that share ``j``: thread ``(lane, j)`` sums ``partials[c, col]``
    for ``c = lane, lane + lanes, ...`` ASCENDING (its residue class), the ``lanes`` per-column partials go through SMEM and row-lane 0
    adds them for ``lane`` ASCENDING, then OVERWRITES ``dw[col]`` (fp32).  Every thread of the block takes the ONE barrier.  The
    chain of a column never depends on ``cols``, so every launch shape over this body is bitwise the same sum (module docstring)."""
    acc = opaque_f32_zero()
    if col < cutlass.Int32(d):
        # Rows in this lane's residue class: lane, lane + lanes, ... < n.
        n_mine = (n - lane + cutlass.Int32(lanes - 1)) // cutlass.Int32(lanes)
        col64 = col.to(cutlass.Int64)
        for k in cutlass.range(n_mine, unroll=unroll):
            c = lane + k * cutlass.Int32(lanes)
            acc = acc + ld_global(p_base + (c.to(cutlass.Int64) * cutlass.Int64(d) + col64) * cutlass.Int64(4), cutlass.Float32)
    sPart.subview(lane * cutlass.Int32(cols) + j).store(acc)
    nvvm.barrier_cta_sync()
    if lane == cutlass.Int32(0):
        if col < cutlass.Int32(d):
            tot = acc
            # lanes ASCENDING, one sequential fp32 chain (the documented order); a
            # fully unrolled dynamic range, not range_constexpr: 127 static
            # iterations trip the DSL's compile-time warning
            for ll in cutlass.range(1, lanes, 1, unroll_full=True):
                tot = tot + sPart.subview(ll * cutlass.Int32(cols) + j).load()
            st_global(o_base + col.to(cutlass.Int64) * cutlass.Int64(4), tot, cutlass.Float32)


@cute.kernel
def frost_dw_norm_reduce(
    mPq: cute.Tensor,  # [n_q, D] fp32 partials
    mPk: cute.Tensor,  # [n_k, D]
    mDWq: cute.Tensor,  # [D] fp32 OUT (overwritten)
    mDWk: cute.Tensor,  # [D]
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    d: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
) -> None:
    """The SECOND launch, class ``blockIdx.y`` (0 = Q, 1 = K), columns
    ``[blockIdx.x * cols, +cols)``: a FIXED summation order, no atomics
    (:func:`dw_reduce_column` per column).

    ``lanes``-way parallel over the rows instead of one thread per column walking
    every row; the shipped ``(cols, lanes) = (8, 128)`` is the measured optimum on
    the SM-fill plane (the table above the module constants). Every partial was
    produced by the preceding launch on the same stream. Host mirror of the order:
    ``test_norm_rope_bwd_fixed_n_ctas_policy_reduces_the_same_sum``.

    SMEM: ``sPart`` fp32 ``lanes x cols`` (4 KiB at (8, 128)), written once per
    thread at ``l * cols + j`` (a warp's 32 threads write 32 consecutive fp32 =
    one 128-B line: conflict-free), read by row-lane 0. Barrier: ONE
    ``barrier_cta_sync`` per block, all ``cols * lanes`` (= 1024) threads arrive
    (no divergent path reaches it). Global reads: a warp covers ``32 / cols``
    rows x ``cols * 4`` B -- whole 32-B sectors at every ``cols >= 8``.
    """
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    j = tidx % cutlass.Int32(cols)
    lane = tidx // cutlass.Int32(cols)
    col = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(cols) + j
    is_q = cutlass.Int32(cute.arch.block_idx()[1]) == cutlass.Int32(0)
    p_base = mPq.iterator.toint() if is_q else mPk.iterator.toint()
    o_base = mDWq.iterator.toint() if is_q else mDWk.iterator.toint()
    n = n_q if is_q else n_k
    sPart = cutlass.Array(cutlass.Float32, lanes * cols, alignment=16, space=cutlass.AddressSpace.smem)
    dw_reduce_column(p_base, o_base, n, col, lane, j, d, cols, lanes, unroll, sPart)


@cute.jit
def qk_norm_rope_bwd_launch(
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    xq: Optional[cute.Tensor],
    xk: Optional[cute.Tensor],
    xk2: Optional[cute.Tensor],
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    pos_q: Optional[cute.Tensor],
    pos_k: Optional[cute.Tensor],
    pos_k2: Optional[cute.Tensor],
    out_q: cute.Tensor,
    out_k: cute.Tensor,
    out_k2: Optional[cute.Tensor],
    out_v: cute.Tensor,
    dw_q: Optional[cute.Tensor],
    dw_k: Optional[cute.Tensor],
    seq_lens: Optional[cute.Tensor],
    amax_out: Optional[cute.Tensor],
    n_ctas_q: cutlass.Int32,
    n_ctas_k: cutlass.Int32,
    n_ctas_v: cutlass.Int32,
    n_q_rows: cutlass.Int32,
    n_k_rows: cutlass.Int32,
    s: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    context_len: cutlass.Int32,
    draft_len: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    batch_one: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qk_norm_rope_bwd(
        dq,
        dk,
        dv,
        xq,
        xk,
        xk2,
        rstd_q,
        rstd_k,
        w_q,
        w_k,
        cos,
        sin,
        pos_q,
        pos_k,
        pos_k2,
        out_q,
        out_k,
        out_k2,
        out_v,
        dw_q,
        dw_k,
        seq_lens,
        amax_out,
        n_ctas_q,
        n_ctas_k,
        n_ctas_v,
        n_q_rows,
        n_k_rows,
        s,
        h_q,
        h_kv,
        context_len,
        draft_len,
        h_q_ct,
        h_kv_ct,
        const_head_counts,
        d,
        rope_dim,
        threads_per_cta,
        rows_per_group,
        batch_one,
        use_pdl,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


@cute.jit
def dw_norm_reduce_launch(
    p_q: cute.Tensor,
    p_k: cute.Tensor,
    dw_q: cute.Tensor,
    dw_k: cute.Tensor,
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_dw_norm_reduce(p_q, p_k, dw_q, dw_k, n_q, n_k, d, cols, lanes, unroll).launch(grid=(n_blocks, 2, 1), block=(cols * lanes, 1, 1), stream=stream)


compiled_cache = {}
reduce_cache = {}


class QkNormRopeBwdRecipe(NamedTuple):
    """Build-time facts of one norm+RoPE backward launch pair.

    Everything here is derivable from the DECLARATION (a legal compile key,
    AGENTS.md Rule 4); the token count enters as a ``cute.sym_int`` and the row
    / CTA counts ride in as runtime ``Int32``. ``n_ctas_cap`` is the per-class
    persistent-CTA cap the policy resolved to at compile time (SM-fill is
    bitwise-stable per device; ``fixed:N`` is cross-device stable).
    ``apply_norm`` / ``want_dw`` / ``has_seq_lens`` record what the artifact
    TRACED; :func:`run_qk_norm_rope_bwd` checks the operands against them in
    BOTH directions (Rule 1). ``eps`` is a recorded FACT of the forward, not a
    kernel input: the backward consumes the forward's SAVED fp32 ``rstd`` and
    never recomputes ``rsqrt(mean(x^2) + eps)``, so ``eps`` is deliberately NOT
    in the compile key (an ``eps`` kernel parameter would be dead)."""

    compiled: object
    reduce_compiled: object
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    rows_per_cta: int
    apply_norm: bool
    want_dw: bool
    has_seq_lens: bool
    n_ctas_cap: int
    dtype: object
    # Appended (defaulted): the rows_per_group the artifact was traced with,
    # after the per-arm default resolution.
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP
    # Appended (default = today's artifact): the amax fold of the stored bands as per-CTA partials (``amax_out`` REQUIRED).
    want_amax: bool = False
    batch_one: bool = False


def _resolve_n_ctas_cap(policy: str, device: int) -> int:
    if policy == "sm_fill":
        return int(_multiprocessor_count(device)) * CTAS_PER_SM
    if policy.startswith("fixed:"):
        try:
            n = int(policy[len("fixed:") :])
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f"n_ctas_policy='fixed:N' needs N >= 1, got {policy!r}")
        return n
    raise ValueError(f"n_ctas_policy must be 'sm_fill' or 'fixed:N', got {policy!r}")



def compile_dspark_qk_norm_rope_bwd(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_counts: bool = DEFAULT_CONST_HEAD_COUNTS,
    batch_one: bool = False,
    use_pdl: bool = False,
    n_ctas_policy: str = DEFAULT_N_CTAS_POLICY,
) -> QkNormRopeBwdRecipe:
    """Compile the DSpark asymmetric, indexed-RoPE training adjoint.

    The logical ``dK`` plane covers ``[context, draft]`` for every batch while
    the source and destination K planes stay split.  The kernel therefore
    consumes the SDPA gradient directly and writes ``dK_context`` /
    ``dK_draft`` directly, without either a forward concat or a backward split.
    """

    global _FAKE_STREAM
    if not isinstance(rows_per_group, int) or rows_per_group < 1:
        raise ValueError(f"rows_per_group must be an int >= 1, got {rows_per_group!r}")
    validate_shape(d, rope_dim, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"DSpark qk_norm_rope_bwd serves bf16/f16 only, got {dtype}")
    device = torch.cuda.current_device()
    n_ctas_cap = _resolve_n_ctas_cap(str(n_ctas_policy), device)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (
        "dspark",
        str(dtype),
        int(h_q),
        int(h_kv),
        int(d),
        int(rope_dim),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_counts),
        bool(batch_one),
        bool(use_pdl),
        str(n_ctas_policy),
        device,
    )
    if key not in compiled_cache:
        tq, tc, td, tk, tp = (cute.sym_int() for _ in range(5))
        dq = _fake(dtype, (tq, h_q, d), (2, 1, 0))
        dk = _fake(dtype, (tk, h_kv, d), (2, 1, 0))
        dv = _fake(dtype, (tk, h_kv, d), (2, 1, 0))
        xq = _fake(dtype, (tq, h_q, d), (2, 1, 0))
        xk_context = _fake(dtype, (tc, h_kv, d), (2, 1, 0))
        xk_draft = _fake(dtype, (td, h_kv, d), (2, 1, 0))
        rstd_q = _fake(torch.float32, (tq, h_q), (1, 0))
        rstd_k = _fake(torch.float32, (tk, h_kv), (1, 0))
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)]
        tables = [_fake(dtype, (tp, rope_dim), (1, 0)) for _ in range(2)]
        positions = [
            _fake(torch.int64, (tq,), (0,)),
            _fake(torch.int64, (tc,), (0,)),
            _fake(torch.int64, (td,), (0,)),
        ]
        out_q = _fake(dtype, (tq, h_q, d), (2, 1, 0))
        out_k_context = _fake(dtype, (tc, h_kv, d), (2, 1, 0))
        out_k_draft = _fake(dtype, (td, h_kv, d), (2, 1, 0))
        out_v_unused = _fake(dtype, (tc, h_kv, d), (2, 1, 0))
        dw_q = _fake(torch.float32, (cute.sym_int(), d), (1, 0))
        dw_k = _fake(torch.float32, (cute.sym_int(), d), (1, 0))
        compiled_cache[key] = cute.compile(
            qk_norm_rope_bwd_launch,
            dq,
            dk,
            dv,
            xq,
            xk_context,
            xk_draft,
            rstd_q,
            rstd_k,
            *weights,
            *tables,
            *positions,
            out_q,
            out_k_context,
            out_k_draft,
            out_v_unused,
            dw_q,
            dw_k,
            None,
            None,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(h_q),
            cutlass.Int32(h_kv),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            int(h_q),
            int(h_kv),
            bool(const_head_counts),
            int(d),
            int(rope_dim),
            int(threads_per_cta),
            int(rows_per_group),
            bool(batch_one),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    reduce_key = (int(d), device)
    if reduce_key not in reduce_cache:
        planes = [_fake(torch.float32, (cute.sym_int(), d), (1, 0)) for _ in range(2)]
        outs32 = [_fake(torch.float32, (d,), (0,)) for _ in range(2)]
        reduce_cache[reduce_key] = cute.compile(
            dw_norm_reduce_launch,
            *planes,
            *outs32,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            int(d),
            REDUCE_COLS,
            REDUCE_LANES,
            REDUCE_UNROLL,
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QkNormRopeBwdRecipe(
        compiled=compiled_cache[key],
        reduce_compiled=reduce_cache[reduce_key],
        h_q=int(h_q),
        h_kv=int(h_kv),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        apply_norm=True,
        want_dw=True,
        has_seq_lens=False,
        n_ctas_cap=int(n_ctas_cap),
        dtype=dtype,
        rows_per_group=int(rows_per_group),
        want_amax=False,
        batch_one=bool(batch_one),
    )


def dspark_n_ctas_for(r: QkNormRopeBwdRecipe, q_tokens: int, k_tokens: int) -> tuple[int, int]:
    """Return the exact Q/K partial-plane row counts for a DSpark launch."""

    counts = []
    for n_rows in (int(q_tokens) * r.h_q, int(k_tokens) * r.h_kv):
        groups = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
        counts.append(max(1, min(groups, r.n_ctas_cap)))
    return counts[0], counts[1]


def _check_fp32_plane(name: str, ten, shape: tuple) -> None:
    want = "[" + ", ".join("*" if x is None else str(x) for x in shape) + "]"
    ok = (
        ten is not None
        and ten.dtype == torch.float32
        and ten.is_contiguous()
        and ten.dim() == len(shape)
        and all(x is None or int(a) == x for a, x in zip(ten.shape, shape))
    )
    if not ok:
        got = "None" if ten is None else f"{ten.dtype} of shape {tuple(ten.shape)} (contiguous={ten.is_contiguous()})"
        raise ValueError(f"{name} must be a contiguous fp32 {want} tensor, got {got}")



def run_dspark_qk_norm_rope_bwd(
    r: QkNormRopeBwdRecipe,
    dq,
    dk,
    xq,
    xk_context,
    xk_draft,
    rstd_q,
    rstd_k,
    w_q,
    w_k,
    cos,
    sin,
    position_q,
    position_context,
    position_draft,
    out_q,
    out_k_context,
    out_k_draft,
    dw_partials_q,
    dw_partials_k,
    dw_q_norm,
    dw_k_norm,
    *,
    batch: int,
    context_len: int,
    draft_len: int,
    stream,
) -> None:
    """Launch the DSpark adjoint and fixed-order norm-weight reduction."""

    if not r.apply_norm or not r.want_dw or r.has_seq_lens or getattr(r, "want_amax", False):
        raise ValueError("run_dspark_qk_norm_rope_bwd needs the DSpark norm-on, dW-on, dense recipe")
    batch = int(batch)
    context_len = int(context_len)
    draft_len = int(draft_len)
    if batch < 1 or context_len < 1 or draft_len < 1:
        raise ValueError(f"batch/context_len/draft_len must be positive, got {(batch, context_len, draft_len)}")
    if r.batch_one and batch != 1:
        raise ValueError(f"this DSpark artifact was specialized for batch=1, got batch={batch}")
    tq = batch * draft_len
    tc = batch * context_len
    td = tq
    tk = batch * (context_len + draft_len)
    expected = (
        ("dq", dq, (tq, r.h_q, r.d)),
        ("dk", dk, (tk, r.h_kv, r.d)),
        ("xq", xq, (tq, r.h_q, r.d)),
        ("xk_context", xk_context, (tc, r.h_kv, r.d)),
        ("xk_draft", xk_draft, (td, r.h_kv, r.d)),
        ("out_q", out_q, (tq, r.h_q, r.d)),
        ("out_k_context", out_k_context, (tc, r.h_kv, r.d)),
        ("out_k_draft", out_k_draft, (td, r.h_kv, r.d)),
    )
    for name, tensor, shape in expected:
        if tensor.dtype != r.dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {r.dtype} with shape {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        if tensor.device != dq.device:
            raise ValueError(f"{name} must be on {dq.device}, got {tensor.device}")
    for name, tensor, n in (
        ("position_q", position_q, tq),
        ("position_context", position_context, tc),
        ("position_draft", position_draft, td),
    ):
        if tensor.dtype != torch.int64 or tuple(tensor.shape) != (n,) or not tensor.is_contiguous() or tensor.device != dq.device:
            raise ValueError(f"{name} must be contiguous int64 [{n}] on {dq.device}")
    if tuple(cos.shape) != tuple(sin.shape) or cos.dtype != r.dtype or sin.dtype != r.dtype or cos.dim() != 2 or int(cos.shape[1]) != r.rope_dim:
        raise ValueError(f"cos/sin must share contiguous {r.dtype} [max_position, {r.rope_dim}] shape")
    if not cos.is_contiguous() or not sin.is_contiguous() or cos.device != dq.device or sin.device != dq.device:
        raise ValueError("cos/sin must be contiguous and on the dQ device")
    for name, tensor, shape, dtype in (
        ("w_q", w_q, (r.d,), r.dtype),
        ("w_k", w_k, (r.d,), r.dtype),
        ("rstd_q", rstd_q, (tq, r.h_q), torch.float32),
        ("rstd_k", rstd_k, (tk, r.h_kv), torch.float32),
        ("dw_q_norm", dw_q_norm, (r.d,), torch.float32),
        ("dw_k_norm", dw_k_norm, (r.d,), torch.float32),
    ):
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.device != dq.device:
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {dq.device}")
    n_ctas_q, n_ctas_k = dspark_n_ctas_for(r, tq, tk)
    _check_fp32_plane("dw_partials_q", dw_partials_q, (n_ctas_q, r.d))
    _check_fp32_plane("dw_partials_k", dw_partials_k, (n_ctas_k, r.d))
    _check_one_cuda_device(
        "dq",
        dq,
        (
            ("dk", dk),
            ("xq", xq),
            ("xk_context", xk_context),
            ("xk_draft", xk_draft),
            ("rstd_q", rstd_q),
            ("rstd_k", rstd_k),
            ("w_q", w_q),
            ("w_k", w_k),
            ("cos", cos),
            ("sin", sin),
            ("position_q", position_q),
            ("position_context", position_context),
            ("position_draft", position_draft),
            ("out_q", out_q),
            ("out_k_context", out_k_context),
            ("out_k_draft", out_k_draft),
            ("dw_partials_q", dw_partials_q),
            ("dw_partials_k", dw_partials_k),
            ("dw_q_norm", dw_q_norm),
            ("dw_k_norm", dw_k_norm),
        ),
    )
    r.compiled(
        dq,
        dk,
        dk,
        xq,
        xk_context,
        xk_draft,
        rstd_q,
        rstd_k,
        w_q,
        w_k,
        cos,
        sin,
        position_q,
        position_context,
        position_draft,
        out_q,
        out_k_context,
        out_k_draft,
        out_k_context,
        dw_partials_q,
        dw_partials_k,
        None,
        None,
        cutlass.Int32(n_ctas_q),
        cutlass.Int32(n_ctas_k),
        cutlass.Int32(0),
        cutlass.Int32(tq * r.h_q),
        cutlass.Int32(tk * r.h_kv),
        cutlass.Int32(0),
        cutlass.Int32(r.h_q),
        cutlass.Int32(r.h_kv),
        cutlass.Int32(context_len),
        cutlass.Int32(draft_len),
        cutlass.Int32(n_ctas_q + n_ctas_k),
        cuda.CUstream(int(stream)),
    )
    run_dw_norm_reduce(r, dw_partials_q, dw_partials_k, dw_q_norm, dw_k_norm, stream=stream)


def run_dw_norm_reduce(
    r: QkNormRopeBwdRecipe,
    dw_partials_q,
    dw_partials_k,
    dw_q_norm,
    dw_k_norm,
    *,
    stream,
) -> None:
    """Reduce per-CTA Q/K norm-weight partials in a fixed order."""
    if not r.want_dw:
        raise ValueError("this artifact was compiled WITHOUT dW partials (want_dw=False); there is nothing to reduce (Rule 1)")
    _check_fp32_plane("dw_partials_q", dw_partials_q, (None, r.d))
    _check_fp32_plane("dw_partials_k", dw_partials_k, (None, r.d))
    _check_fp32_plane("dw_q_norm", dw_q_norm, (r.d,))
    _check_fp32_plane("dw_k_norm", dw_k_norm, (r.d,))
    n_q, n_k = int(dw_partials_q.shape[0]), int(dw_partials_k.shape[0])
    if n_q < 1 or n_k < 1:
        raise ValueError(f"the partial planes need at least one row each, got {n_q} / {n_k}")
    _check_one_cuda_device("dw_partials_q", dw_partials_q, (("dw_partials_k", dw_partials_k), ("dw_q_norm", dw_q_norm), ("dw_k_norm", dw_k_norm)))
    n_blocks = (r.d + REDUCE_COLS - 1) // REDUCE_COLS
    r.reduce_compiled(
        dw_partials_q, dw_partials_k, dw_q_norm, dw_k_norm, cutlass.Int32(n_q), cutlass.Int32(n_k), cutlass.Int32(n_blocks), cuda.CUstream(int(stream))
    )


# Pin profiler-visible symbols. ptxas scheduling can change when a kernel's
# symbol changes, so this is part of the performance contract of the vendored
# specialization rather than cosmetic naming.
frost_qk_norm_rope_bwd.set_name_prefix("aspec", remove_cutlass_symbol=True)
frost_dw_norm_reduce.set_name_prefix("aspec", remove_cutlass_symbol=True)
