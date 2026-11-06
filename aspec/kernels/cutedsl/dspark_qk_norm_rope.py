# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted by ASpec Authors from NVIDIA cuDNN Frontend cb8cec8.

"""ASpec-owned CuTeDSL kernel for DSpark Q/K RMSNorm and indexed RoPE.

The kernel reads draft Q plus separate context/draft K planes and writes compact
Q and logical ``[context, draft]`` K outputs without materializing a K concat.
RMSNorm and RoPE arithmetic stay in FP32 and round once to the input dtype.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental import primitives as nvvm

from aspec.kernels.cutedsl._primitives import (
    dtype_to_cutlass,
    f16x2_to_f32,
    fp32_to_fp16,
    lane_group_sum,
    launch_dependent_grids,
    ld_global,
    ld_global_v4,
    st_global,
    st_global_v2,
    st_global_v4,
    wait_on_dependent_grids,
)

COPY_BITS = 128
ELEMS_PER_ACCESS = COPY_BITS // 16  # halves moved by one ld.global.v4
WORDS_PER_ACCESS = COPY_BITS // 32
ACCESS_BYTES = COPY_BITS // 8
DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_DEFER_SECONDARY_LOADS = True
DEFAULT_CONST_HEAD_COUNTS = True
# Head counts are already part of the compile key. Treating them as constants
# removes runtime integer division without changing address or arithmetic semantics.
# Deferred secondary loads keep norm weights and RoPE values out of the live set
# during the sum-of-squares pass; this is the main register-pressure optimization.
# The coupled thread/row defaults are validated per architecture and output mode.
# Numerical comparisons and supported test cases are maintained in
# tests/test_dspark_qk_cutedsl.py.

_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)


def lanes_per_row(d: int) -> int:
    """Lanes that cooperate on one head row: enough for one 16-byte access each,
    capped at a warp so a row never straddles a shuffle group."""
    return min(32, d // ELEMS_PER_ACCESS)


def vec_chunks(d: int) -> int:
    """Accesses each lane makes per row."""
    return d // (lanes_per_row(d) * ELEMS_PER_ACCESS)


def validate_shape(d: int, rope_dim: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address.

    Never an ``assert``: these come from user-facing geometry, so they must
    survive ``python -O`` and name themselves (engine contract § 7).
    """
    if d % ELEMS_PER_ACCESS != 0:
        raise ValueError(f"d_head must be a multiple of {ELEMS_PER_ACCESS} for {COPY_BITS}-bit accesses, got {d}")
    lanes = lanes_per_row(d)
    if 32 % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp so the reduction and the RoPE shuffle stay in-row")
    if d != lanes * ELEMS_PER_ACCESS * vec_chunks(d):
        raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {vec_chunks(d)} accesses")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")
    if rope_dim:
        if rope_dim % (2 * ELEMS_PER_ACCESS) != 0:
            raise ValueError(f"rope_dim must be a multiple of {2 * ELEMS_PER_ACCESS} so the rotate_half partner is a whole-lane shuffle, got {rope_dim}")
        half_lanes = rope_dim // (2 * ELEMS_PER_ACCESS)
        if half_lanes & (half_lanes - 1):
            raise ValueError(f"rope_dim/{2 * ELEMS_PER_ACCESS} must be a power of two for the butterfly shuffle, got {half_lanes}")
        if rope_dim > lanes * ELEMS_PER_ACCESS:
            raise ValueError(
                f"rope_dim={rope_dim} must fit in the first access chunk ({lanes * ELEMS_PER_ACCESS} elements) so the shuffle needs no cross-chunk exchange"
            )


@cute.kernel

def frost_qk_norm_rope(
    mQ: cute.Tensor,  # [T, H_q,  D] in
    mK: cute.Tensor,  # [T, H_kv, D] in
    mK2: Optional[cute.Tensor],  # optional second K source (DSpark draft K)
    mQo: cute.Tensor,  # [T, H_q,  D] out (may alias mQ)
    mKo: cute.Tensor,  # [T, H_kv, D] out (may alias mK)
    mWq: Optional[cute.Tensor],  # [D], or None: RoPE-only (no RMSNorm, no weight loads, no rstd)
    mWk: Optional[cute.Tensor],  # [D], or None (both or neither -- the host checks)
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,  # [T, ROPE_DIM]
    mPosQ: Optional[cute.Tensor],  # optional int64 positions for Q
    mPosK: Optional[cute.Tensor],  # optional int64 positions for the first K source
    mPosK2: Optional[cute.Tensor],  # optional int64 positions for the second K source
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32, or None
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32, or None
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    eps: cutlass.Float32,
    context_len: cutlass.Int32,
    draft_len: cutlass.Int32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    defer_secondary_loads: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    batch_one: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``rows_per_group`` consecutive rows per lane group: ALL loads issued
    first, then the reductions and stores.

    That split is the kernel's whole latency story. One row is a single 16-byte
    load per lane, and the sum-of-squares depends on it immediately — so at
    ``rows_per_group == 1`` each thread has exactly one request in flight and
    the kernel runs at whatever memory-level parallelism occupancy alone
    provides. Issuing R independent rows first multiplies that by R at the cost
    of R times the live registers.

    The two norm weights are loaded ONCE per thread, not once per row: they are
    ``[D]`` and every row picks one of the two with a register select.

    Tail rows clamp their loads to the last valid row and skip every store, so
    the ragged final CTA costs a redundant read and never a wild write.

    ``apply_norm`` is decided at TRACE time from the presence of the weight
    tensors (the same presence switch ``want_rstd`` uses for the rstd outputs):
    with ``mWq is None`` the sum-of-squares pass, the rsqrt, the weight loads
    and the rstd store are not traced at all, and the kernel is partial RoPE
    on Q/K with the dims ``[rope_dim, D)`` copied through BIT-EXACTLY (they
    never leave the io dtype: no fp32 op touches them).
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    apply_norm = cutlass.const_expr(mWq is not None)
    split_k = cutlass.const_expr(mK2 is not None)
    indexed_rope = cutlass.const_expr(mPosQ is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)

    # Address-math folding. Every one of these is a compile-time fact of the
    # artifact that the shipped signature was carrying as a RUNTIME value, and
    # this kernel is register-and-address bound, so each one is real:
    #
    # * ``h_q``/``h_kv`` are in the compile cache key, yet ``row // h_q`` was an
    #   integer DIVISION by a runtime register. Folded, it is a shift.
    # * Q, K, GATE and V are column slices of ONE fused projection, so they
    #   SHARE a token stride by construction (the section-1 layout contract).
    #   Four symbolic strides the compiler cannot prove equal become one.
    # * the block writes COMPACT Q_c/K_c, so the output token stride is
    #   ``h * d`` and need not be symbolic at all.
    _hq = cutlass.Int32(h_q_ct) if cutlass.const_expr(const_head_counts) else h_q
    _hkv = cutlass.Int32(h_kv_ct) if cutlass.const_expr(const_head_counts) else h_kv

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)

    # The norm weight is picked by ADDRESS, then loaded once per row. Loading
    # both [D] vectors up front and selecting in registers looks like it saves
    # a load, but it pins 2x chunks*8 live fp32 registers for the whole kernel
    # -- and this kernel is occupancy-bound, not issue-bound (measured on
    # Rubin: rows_per_group=1 beats 2 beats 4, i.e. more per-thread state makes
    # it strictly worse). The extra load is an L1 hit; the registers are not free.

    rope_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    in_rope = lane < cutlass.Int32(rope_lanes) if cutlass.const_expr(rope_dim > 0) else cutlass.Boolean(False)

    # --- PASS 1: issue every load this thread will make ----------------------
    rows = []
    dsts = []
    rstds = []
    is_qs = []
    tokens = []
    xs = []
    ws = []
    cs = []
    ss = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        is_q = row_r < n_q_rows
        src_addr = cutlass.Int64(0)
        dst_addr = cutlass.Int64(0)
        rstd_addr = cutlass.Int64(0)
        token = cutlass.Int32(0)
        source_token = cutlass.Int32(0)
        pos_base = cutlass.Int64(0)
        if is_q:
            token = row_r // _hq
            source_token = token
            head = row_r % _hq
            src_addr = mQ.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mQ.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mQ.stride[1])
            ) * cutlass.Int64(2)
            dst_addr = mQo.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mQo.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mQo.stride[1])
            ) * cutlass.Int64(2)
            if cutlass.const_expr(mRstdQ is not None):
                rstd_addr = mRstdQ.iterator.toint() + row_r.to(cutlass.Int64) * cutlass.Int64(4)
            if cutlass.const_expr(indexed_rope):
                pos_base = mPosQ.iterator.toint()
        else:
            k_row = row_r - n_q_rows
            token = k_row // _hkv
            head = k_row % _hkv
            source_token = token
            src_base = mK.iterator.toint()
            src_tok_stride = cutlass.Int64(mK.stride[0])
            src_head_stride = cutlass.Int64(mK.stride[1])
            if cutlass.const_expr(split_k):
                total_len = context_len + draft_len
                if cutlass.const_expr(batch_one):
                    bidx = cutlass.Int32(0)
                    pos = token
                else:
                    bidx = token // total_len
                    pos = token - bidx * total_len
                if pos < context_len:
                    source_token = bidx * context_len + pos
                    if cutlass.const_expr(indexed_rope):
                        pos_base = mPosK.iterator.toint()
                else:
                    source_token = bidx * draft_len + (pos - context_len)
                    src_base = mK2.iterator.toint()
                    src_tok_stride = cutlass.Int64(mK2.stride[0])
                    src_head_stride = cutlass.Int64(mK2.stride[1])
                    if cutlass.const_expr(indexed_rope):
                        pos_base = mPosK2.iterator.toint()
            else:
                if cutlass.const_expr(indexed_rope):
                    pos_base = mPosK.iterator.toint()
            src_addr = src_base + (source_token.to(cutlass.Int64) * src_tok_stride + head.to(cutlass.Int64) * src_head_stride) * cutlass.Int64(2)
            dst_addr = mKo.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mKo.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mKo.stride[1])
            ) * cutlass.Int64(2)
            if cutlass.const_expr(mRstdK is not None):
                rstd_addr = mRstdK.iterator.toint() + k_row.to(cutlass.Int64) * cutlass.Int64(4)

        rope_token = token
        if cutlass.const_expr(indexed_rope):
            rope_token = ld_global(pos_base + source_token.to(cutlass.Int64) * cutlass.Int64(8), cutlass.Int64).to(cutlass.Int32)

        # The weight address exists only when there IS a weight: under
        # apply_norm=False no line below reads mWq / mWk.
        w_addr = cutlass.Int64(0)
        if cutlass.const_expr(apply_norm):
            w_addr = mWq.iterator.toint() if is_q else mWk.iterator.toint()
        row_x = []
        row_w = []
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
            pairs = [f16x2_to_f32(w, dtype=mQ.element_type) for w in ld_global_v4(src_addr + off, cutlass.Int32)]
            row_x.append([v for pair in pairs for v in pair])
            if cutlass.const_expr(apply_norm and not defer_secondary_loads):
                w_pairs = [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr + off, cutlass.Int32)]
                row_w.append([v for pair in w_pairs for v in pair])

        # cos/sin only on the rope lanes -- rope_dim/D of the warp. Loading them
        # on every lane (a clamped read) tripled this kernel's L1 request count
        # for no data, and cost real bandwidth.
        cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        if cutlass.const_expr(rope_dim > 0 and not defer_secondary_loads):
            if in_rope:
                base = mCos.iterator.toint() + rope_token.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(2) + rope_off
                base_s = mSin.iterator.toint() + rope_token.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(2) + rope_off
                c_pairs = [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(base, cutlass.Int32)]
                s_pairs = [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(base_s, cutlass.Int32)]
                cos_v = [v for pair in c_pairs for v in pair]
                sin_v = [v for pair in s_pairs for v in pair]

        rows.append(row)
        dsts.append(dst_addr)
        rstds.append(rstd_addr)
        is_qs.append(is_q)
        tokens.append(rope_token)
        xs.append(row_x)
        ws.append(row_w)
        cs.append(cos_v)
        ss.append(sin_v)

    # --- PASS 2: reduce, normalize, rotate, store ----------------------------
    # rstd exists only where a norm exists: compile_qk_norm_rope refuses
    # want_rstd without apply_norm, so this conjunction never folds a store out
    # behind a caller's back -- it only keeps the kernel self-consistent.
    want_rstd = cutlass.const_expr(apply_norm and (mRstdQ is not None or mRstdK is not None))
    rstd_vals = []
    for r in cutlass.range_constexpr(rows_per_group):
        # Pre-bound so the name exists on both trace paths; the value is read
        # only under apply_norm, where it is replaced by the real rsqrt.
        rstd = cutlass.Float32(1.0)
        if cutlass.const_expr(apply_norm):
            acc = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(chunks):
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = cute.math.fma(xs[r][c][i], xs[r][c][i], acc)
            rstd = cute.math.rsqrt(lane_group_sum(acc, lanes) * cutlass.Float32(1.0 / d) + eps, fastmath=True)

        # The norm weight and the cos/sin pair are consumed HERE, after the
        # reduction. Loading them in PASS 1 costs 24 live fp32 per row across
        # the whole sum-of-squares tree, and this kernel is REGISTER-PRESSURE
        # bound, not request-starved: measured cold on Rubin, three unrelated
        # register reductions each buy 20-25% while every knob that adds rows
        # in flight makes it strictly worse. So the early issue buys
        # memory-level parallelism the kernel does not need, with the one
        # resource it does. Both are L1/L2 hits by construction -- the weight
        # is [D] and shared by every row, cos/sin are shared by every head of a
        # token -- so deferring them exposes almost no latency.
        if cutlass.const_expr(defer_secondary_loads):
            if cutlass.const_expr(apply_norm):
                w_addr_r = mWq.iterator.toint() if is_qs[r] else mWk.iterator.toint()
                row_w = []
                for c in cutlass.range_constexpr(chunks):
                    off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
                    w_pairs = [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr_r + off, cutlass.Int32)]
                    row_w.append([v for pair in w_pairs for v in pair])
                ws[r] = row_w
            if cutlass.const_expr(rope_dim > 0):
                if in_rope:
                    tok = tokens[r]
                    base = mCos.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) * cutlass.Int64(2) + rope_off
                    base_s = mSin.iterator.toint() + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) * cutlass.Int64(2) + rope_off
                    c_pairs = [f16x2_to_f32(w, dtype=mCos.element_type) for w in ld_global_v4(base, cutlass.Int32)]
                    s_pairs = [f16x2_to_f32(w, dtype=mSin.element_type) for w in ld_global_v4(base_s, cutlass.Int32)]
                    cs[r] = [v for pair in c_pairs for v in pair]
                    ss[r] = [v for pair in s_pairs for v in pair]

        # RoPE-only: ys IS xs. The passthrough dims [rope_dim, D) then go back
        # through fp32_to_fp16 untouched by any fp32 op, which is exactly a
        # widen + round-to-same -> bit-exact copy (asserted by the tests).
        ys = []
        for c in cutlass.range_constexpr(chunks):
            if cutlass.const_expr(apply_norm):
                ys.append([xs[r][c][i] * rstd * ws[r][c][i] for i in range(ELEMS_PER_ACCESS)])
            else:
                ys.append(list(xs[r][c]))

        if cutlass.const_expr(rope_dim > 0):
            # rotate_half: element e pairs with e + rope_dim/2, the SAME
            # sub-index in lane (l ^ half_lanes). The shuffle is unconditional
            # -- shfl.sync must be reached by every lane of the warp -- and only
            # the rope lanes keep the result.
            rotated = []
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, ys[0][i], cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                signed = -partner if lane < cutlass.Int32(half_lanes) else partner
                rotated.append(cute.math.fma(ys[0][i], cs[r][i], signed * ss[r][i]))
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                ys[0][i] = rotated[i] if in_rope else ys[0][i]

        if cutlass.const_expr(apply_norm):
            rstd_vals.append(rstd)
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
                y = ys[c]
                packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mQo.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                st_global_v4(dsts[r] + off, packed, cutlass.Int32)

    # --- rstd, stored as ONE vector when the group is contiguous -------------
    # One lane per row storing 4 scattered bytes turns each row into its own
    # 32-byte sector: measured 86.6% -> 72.5% of the HBM ceiling at
    # rows_per_group=2 on an A100. A group's R rows ARE consecutive in the same
    # rstd tensor unless it straddles the Q/K seam or the ragged tail, so check
    # that at run time and vectorize the common case.
    #
    # Contiguous is NECESSARY, not sufficient: a v2/v4 store also needs its
    # address R*4-byte ALIGNED. ``row0`` is a multiple of R, so the Q side
    # (``mRstdQ + row0*4``) always is -- but the K side lands at
    # ``mRstdK + (row0 - n_q_rows)*4``, which is aligned only when ``n_q_rows =
    # T*h_q`` is itself a multiple of R. It is not at T=3, h_q=3, R=2 (n_q_rows=9:
    # the group at rows 10..11 is whole-K and its K offset is 4 B), and the
    # vector store then faults with ``cudaErrorMisalignedAddress`` (found by
    # review on PR #1102). So a whole-K group additionally requires the aligned
    # K offset; every other group takes the scalar path below, which never
    # needed more than 4-byte alignment.
    if cutlass.const_expr(want_rstd):
        if lane == cutlass.Int32(0):
            if cutlass.const_expr(rows_per_group in (2, 4)):
                whole_q = row0 + cutlass.Int32(rows_per_group - 1) < n_q_rows
                k_aligned = (row0 - n_q_rows) % cutlass.Int32(rows_per_group) == cutlass.Int32(0)
                whole_k = (row0 >= n_q_rows) and k_aligned
                if (whole_q or whole_k) and (row0 + cutlass.Int32(rows_per_group) <= n_rows):
                    if cutlass.const_expr(rows_per_group == 2):
                        st_global_v2(rstds[0], rstd_vals, cutlass.Float32)
                    else:
                        st_global_v4(rstds[0], rstd_vals, cutlass.Float32)
                else:
                    for r in cutlass.range_constexpr(rows_per_group):
                        if rows[r] < n_rows:
                            st_global(rstds[r], rstd_vals[r], cutlass.Float32)
            else:
                for r in cutlass.range_constexpr(rows_per_group):
                    if rows[r] < n_rows:
                        st_global(rstds[r], rstd_vals[r], cutlass.Float32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qk_norm_rope_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    k2: Optional[cute.Tensor],
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    pos_q: Optional[cute.Tensor],
    pos_k: Optional[cute.Tensor],
    pos_k2: Optional[cute.Tensor],
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    eps: cutlass.Float32,
    context_len: cutlass.Int32,
    draft_len: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    defer_secondary_loads: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    batch_one: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qk_norm_rope(
        q,
        k,
        k2,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        pos_q,
        pos_k,
        pos_k2,
        rstd_q,
        rstd_k,
        n_q_rows,
        n_rows,
        h_q,
        h_kv,
        eps,
        context_len,
        draft_len,
        d,
        rope_dim,
        threads_per_cta,
        rows_per_group,
        defer_secondary_loads,
        h_q_ct,
        h_kv_ct,
        const_head_counts,
        batch_one,
        use_pdl,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}

class DSparkQkNormRopeRecipe(NamedTuple):
    """Compiled DSpark Q/K RMSNorm+RoPE forward with asymmetric token planes.

    Q covers the draft tokens.  K is read directly from separate context and
    draft projections and written in logical ``[context, draft]`` order per
    batch, so execute does not materialize a concatenated pre-norm K tensor.
    Sequence extents are runtime values and never enter the compile key.
    """

    compiled: object
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    rows_per_cta: int
    batch_one: bool = False


def _fake(dtype, shape, stride_order):
    return cute.runtime.make_fake_compact_tensor(
        dtype=dtype_to_cutlass(dtype),
        shape=shape,
        stride_order=stride_order,
        assumed_align=16,
    )



def compile_dspark_qk_norm_rope(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = 1,
    defer_secondary_loads: bool = DEFAULT_DEFER_SECONDARY_LOADS,
    const_head_counts: bool = DEFAULT_CONST_HEAD_COUNTS,
    batch_one: bool = False,
    use_pdl: bool = False,
) -> DSparkQkNormRopeRecipe:
    """Compile the DSpark asymmetric, indexed-RoPE training specialization."""

    validate_shape(d, rope_dim, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"DSpark qk_norm_rope serves bf16/f16 only, got {dtype}")
    key = (
        "dspark",
        str(dtype),
        int(h_q),
        int(h_kv),
        int(d),
        int(rope_dim),
        int(threads_per_cta),
        int(rows_per_group),
        bool(defer_secondary_loads),
        bool(const_head_counts),
        bool(batch_one),
        bool(use_pdl),
        torch.cuda.current_device(),
    )
    if key not in compiled_cache:
        tq, tc, td, tk, tp = (cute.sym_int() for _ in range(5))
        q = _fake(dtype, (tq, h_q, d), (2, 1, 0))
        k_context = _fake(dtype, (tc, h_kv, d), (2, 1, 0))
        k_draft = _fake(dtype, (td, h_kv, d), (2, 1, 0))
        q_out = _fake(dtype, (tq, h_q, d), (2, 1, 0))
        k_out = _fake(dtype, (tk, h_kv, d), (2, 1, 0))
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)]
        tables = [_fake(dtype, (tp, rope_dim), (1, 0)) for _ in range(2)]
        positions = [
            _fake(torch.int64, (tq,), (0,)),
            _fake(torch.int64, (tc,), (0,)),
            _fake(torch.int64, (td,), (0,)),
        ]
        rstd_q = _fake(torch.float32, (tq, h_q), (1, 0))
        rstd_k = _fake(torch.float32, (tk, h_kv), (1, 0))
        compiled_cache[key] = cute.compile(
            qk_norm_rope_launch,
            q,
            k_context,
            k_draft,
            q_out,
            k_out,
            *weights,
            *tables,
            *positions,
            rstd_q,
            rstd_k,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(h_q),
            cutlass.Int32(h_kv),
            cutlass.Float32(eps),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            d,
            int(rope_dim),
            int(threads_per_cta),
            int(rows_per_group),
            bool(defer_secondary_loads),
            int(h_q),
            int(h_kv),
            bool(const_head_counts),
            bool(batch_one),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return DSparkQkNormRopeRecipe(
        compiled=compiled_cache[key],
        h_q=int(h_q),
        h_kv=int(h_kv),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        batch_one=bool(batch_one),
    )


def run_dspark_qk_norm_rope(
    r: DSparkQkNormRopeRecipe,
    q,
    k_context,
    k_draft,
    q_out,
    k_out,
    w_q,
    w_k,
    cos,
    sin,
    position_q,
    position_context,
    position_draft,
    rstd_q,
    rstd_k,
    *,
    batch: int,
    context_len: int,
    draft_len: int,
    stream,
) -> None:
    """Launch a precompiled DSpark Q/K preparation artifact without allocation."""

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
        ("q", q, (tq, r.h_q, r.d)),
        ("k_context", k_context, (tc, r.h_kv, r.d)),
        ("k_draft", k_draft, (td, r.h_kv, r.d)),
        ("q_out", q_out, (tq, r.h_q, r.d)),
        ("k_out", k_out, (tk, r.h_kv, r.d)),
    )
    for name, tensor, shape in expected:
        if tensor.dtype != q.dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {q.dtype} with shape {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
    for name, tensor, n in (
        ("position_q", position_q, tq),
        ("position_context", position_context, tc),
        ("position_draft", position_draft, td),
    ):
        if tensor.dtype != torch.int64 or tuple(tensor.shape) != (n,) or not tensor.is_contiguous() or tensor.device != q.device:
            raise ValueError(f"{name} must be contiguous int64 [{n}] on {q.device}")
    if tuple(cos.shape) != tuple(sin.shape) or cos.dtype != q.dtype or sin.dtype != q.dtype or cos.dim() != 2 or int(cos.shape[1]) != r.rope_dim:
        raise ValueError(f"cos/sin must share contiguous {q.dtype} [max_position, {r.rope_dim}] shape")
    if not cos.is_contiguous() or not sin.is_contiguous() or cos.device != q.device or sin.device != q.device:
        raise ValueError("cos/sin must be contiguous and on the Q device")
    for name, tensor, shape, dtype in (
        ("w_q", w_q, (r.d,), q.dtype),
        ("w_k", w_k, (r.d,), q.dtype),
        ("rstd_q", rstd_q, (tq, r.h_q), torch.float32),
        ("rstd_k", rstd_k, (tk, r.h_kv), torch.float32),
    ):
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.device != q.device:
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {q.device}")
    n_q_rows = tq * r.h_q
    n_rows = n_q_rows + tk * r.h_kv
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(
        q,
        k_context,
        k_draft,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        position_q,
        position_context,
        position_draft,
        rstd_q,
        rstd_k,
        cutlass.Int32(n_q_rows),
        cutlass.Int32(n_rows),
        cutlass.Int32(r.h_q),
        cutlass.Int32(r.h_kv),
        cutlass.Float32(r.eps),
        cutlass.Int32(context_len),
        cutlass.Int32(draft_len),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


# Keep the generated symbol stable and independent of CuTeDSL's default
# ``kernel_cutlass`` prefix. Besides making profiler output attributable to
# ASpec, the stable prefix avoids ptxas schedule drift across package moves.
frost_qk_norm_rope.set_name_prefix("aspec", remove_cutlass_symbol=True)
