# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in folded/native GQA checks, including differentiable CP LSE output."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires CUDA cuDNN GQA layout integration",
)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_folded_gqa_matches_native_output_lse_and_all_gradients(thd, ratio):
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(924)
    batch, queries, keys, kv_heads, dim = 2, 43, 179, 2, 128
    heads = kv_heads * ratio
    data = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in
            ((batch, queries, heads, dim), (batch, keys, kv_heads, dim), (batch, keys, kv_heads, dim))]
    lengths = [37, 43] if thd else [queries, queries]
    key_lengths = [163, 179] if thd else [keys, keys]
    if thd:
        data = [torch.cat([value[0, :lengths[0] if i == 0 else key_lengths[0]], value[1]])
                for i, value in enumerate(data)]
    endpoints = []
    for qlen, klen in zip(lengths, key_lengths, strict=True):
        ends = torch.arange(qlen, device="cuda") % (klen - 9) + 1
        row = torch.stack((torch.zeros_like(ends), torch.zeros_like(ends), ends, ends + 3, ends + 7))
        row[:, 11] = 0  # Empty visibility must preserve -inf LSE and zero output.
        endpoints.append(row)
    mask = CudnnFlexMask(torch.cat(endpoints, dim=-1).int()[None].contiguous(), pack_gqa=False,
                         **(dict(cu_seqlens_q=torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
                                 cu_seqlens_k=torch.tensor([0, key_lengths[0], sum(key_lengths)], device="cuda", dtype=torch.int32),
                                 max_seqlen_q=max(lengths), max_seqlen_k=max(key_lengths)) if thd else {}))
    dout = torch.randn_like(data[0])  # Keep the actual attention dO boundary BF16.
    lse_shape = (heads, sum(lengths)) if thd else (batch, heads, queries)
    dlse = torch.randn(lse_shape, device="cuda", dtype=torch.float32) * 0.01
    results = []
    for fold in (False, True):
        q, k, v = [x.clone().requires_grad_() for x in data]
        output, lse = cudnn_flex_attention(q, k, v, mask=mask, scale=dim ** -0.5,
                                          physical_gqa=fold, return_lse=True)
        assert output.shape == q.shape and lse.shape == lse_shape
        gradients = torch.autograd.grad((output, lse), (q, k, v), (dout, dlse))
        assert all(torch.isfinite(g).all() for g in gradients)
        results.append((output, lse, gradients))
    for actual, expected in zip(results[1][:2], results[0][:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.001)
    for actual, expected in zip(results[1][2], results[0][2], strict=True):
        relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp(min=1e-20)
        assert relative < 0.01, relative.item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_folded_thd_attention_without_lse_preserves_return_contract(dtype):
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    q = torch.randn(21, 8, 128, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(73, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = CudnnFlexMask(torch.full((1, 1, 21), 73, device="cuda", dtype=torch.int32),
                         cu_seqlens_q=torch.tensor([0, 21], device="cuda", dtype=torch.int32),
                         cu_seqlens_k=torch.tensor([0, 73], device="cuda", dtype=torch.int32),
                         max_seqlen_q=21, max_seqlen_k=73)
    output = cudnn_flex_attention(q, k, v, mask=mask, scale=128 ** -0.5, physical_gqa=True)
    assert isinstance(output, torch.Tensor) and output.shape == q.shape
    # Match the backend's native aligned dO contract. sum() would instead send
    # a stride-zero expanded gradient, rejected by native cuDNN as well.
    gradients = torch.autograd.grad(output, (q, k, v), torch.randn_like(output))
    assert all(torch.isfinite(g).all() for g in gradients)
