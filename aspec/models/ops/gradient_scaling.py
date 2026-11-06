# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pointwise gradient normalization with the native CUDA scalar rounding.

Reimplemented from PyTorch's in-place tensor multiplication contract. No
reduction, optimizer state or normalization denominator moves into this kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_bf16(gradient, coefficient, length, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(gradient + index, index < length, 0).to(tl.float32)
    # A CUDA zero-dimensional tensor follows tensor dtype promotion, unlike a
    # Python scalar. Keep the BF16 coefficient rounding before FP32 opmath.
    scale = tl.load(coefficient).to(gradient.dtype.element_ty).to(tl.float32)
    tl.store(gradient + index, value * scale, index < length)


def scale_gradients_(gradients, coefficient):
    """Scale in place; preserve native handling of DTensor and other layouts."""
    use_kernel = (
        not torch.compiler.is_compiling()
        and type(coefficient) is torch.Tensor
        and coefficient.is_cuda
        and coefficient.ndim == 0
        and coefficient.dtype == torch.float32
        and torch.cuda.get_device_capability(coefficient.device) == (9, 0)
    )
    for gradient in gradients:
        if (
            use_kernel
            and type(gradient) is torch.Tensor
            and gradient.device == coefficient.device
            and gradient.dtype == torch.bfloat16
            and gradient.is_contiguous()
        ):
            if gradient.numel():
                _scale_bf16[(triton.cdiv(gradient.numel(), 1024),)](
                    gradient, coefficient, gradient.numel(), BLOCK=1024
                )
        else:
            gradient.mul_(coefficient)
