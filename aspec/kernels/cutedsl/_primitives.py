# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Adapted by ASpec Authors from NVIDIA cuDNN Frontend cb8cec8.

"""Small CuTeDSL primitives used by ASpec kernels.

Keeping this surface local avoids a runtime dependency on cuDNN Frontend's
private ``cudnn.frost`` modules.  These helpers intentionally expose only the
32/64/128-bit accesses and reductions required by the DSpark Q/K kernel.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm


def dtype_to_cutlass(dtype: torch.dtype):
    """Map the tensor dtypes accepted by this kernel to CuTeDSL types."""

    mapping = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported CuTeDSL tensor dtype: {dtype}") from exc


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype=cutlass.Float16):
    """Pack two FP32 values into one FP16/BF16 word with round-to-nearest."""

    if cutlass.const_expr(dtype not in (cutlass.Float16, cutlass.BFloat16)):
        raise TypeError(f"dtype must be Float16 or BFloat16, got {dtype}")
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    return inline_ptx(
        f"cvt.rn.{tag}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


@cute.jit
def f16x2_to_f32(word, *, dtype=cutlass.Float16):
    """Unpack one FP16/BF16 word into two FP32 values."""

    if cutlass.const_expr(dtype not in (cutlass.Float16, cutlass.BFloat16)):
        raise TypeError(f"dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        return inline_ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; "
            "mov.b32 $0, {0, l}; mov.b32 $1, {0, h}; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    return inline_ptx(
        "{ .reg .b16 h0, h1; mov.b32 {h0, h1}, $2; "
        "cvt.f32.f16 $0, h0; cvt.f32.f16 $1, h1; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[word],
    )


@cute.jit
def lane_group_sum(value: cutlass.Float32, lanes: cutlass.Constexpr[int]):
    """Sum across a power-of-two group of consecutive lanes."""

    offset = lanes // 2
    while offset >= 1:
        value = value + cutlass.Float32(
            nvvm.shfl_sync(0xFFFFFFFF, value, offset, 31, kind=nvvm.Shfl.BFLY)
        )
        offset //= 2
    return value


@cute.jit
def opaque_f32_zero():
    """Return zero in a register so inline-PTX operands cannot become literals."""

    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Float32])


@cute.jit
def fmax_f32(a: cutlass.Float32, b: cutlass.Float32):
    return inline_ptx(
        "max.f32 $0, $1, $2;",
        write_only_types=[cutlass.Float32],
        read_only_args=[a, b],
    )


def abs_max_tree(values):
    """Return ``max(abs(values))`` as a shallow ternary tree."""

    if not values:
        raise ValueError("abs_max_tree needs at least one value")
    nodes = [cute.math.abs(value) for value in values]
    while len(nodes) > 1:
        reduced = []
        for start in range(0, len(nodes), 3):
            group = nodes[start : start + 3]
            value = group[0]
            for other in group[1:]:
                value = fmax_f32(value, other)
            reduced.append(value)
        nodes = reduced
    return nodes[0]


@cute.jit
def warp_abs_max_f32_shfl(value: cutlass.Float32):
    value = cute.math.abs(value)
    for index in cutlass.range_constexpr(5):
        value = fmax_f32(value, cute.arch.shuffle_sync_bfly(value, 1 << index))
    return value


def _ptx_type_suffix(dtype) -> str:
    return "f32" if dtype == cutlass.Float32 else "b32"


def ld_global(address, dtype):
    return nvvm.inline_ptx(
        f"ld.global.{_ptx_type_suffix(dtype)} $0, [$1];",
        write_only_types=[dtype],
        read_only_args=[address],
    )


def ld_global_v4(address, dtype):
    return nvvm.inline_ptx(
        f"ld.global.v4.{_ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[address],
    )


def st_global(address, value, dtype):
    nvvm.inline_ptx(
        f"st.global.{_ptx_type_suffix(dtype)} [$0], $1;",
        read_only_args=[address, value],
    )


def st_global_v2(address, values, dtype):
    nvvm.inline_ptx(
        f"st.global.v2.{_ptx_type_suffix(dtype)} [$0], {{$1, $2}};",
        read_only_args=[address, values[0], values[1]],
    )


def st_global_v4(address, values, dtype):
    nvvm.inline_ptx(
        f"st.global.v4.{_ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[address, values[0], values[1], values[2], values[3]],
    )


def ld_shared_v4(address, dtype):
    return nvvm.inline_ptx(
        f"ld.shared.v4.{_ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[address],
    )


def st_shared_v4(address, values, dtype):
    nvvm.inline_ptx(
        f"st.shared.v4.{_ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[address, values[0], values[1], values[2], values[3]],
    )


@cute.jit
def wait_on_dependent_grids():
    inline_ptx("griddepcontrol.wait;", write_only_types=[], read_only_args=[])


@cute.jit
def launch_dependent_grids():
    inline_ptx(
        "griddepcontrol.launch_dependents;",
        write_only_types=[],
        read_only_args=[],
    )
