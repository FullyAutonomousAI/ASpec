# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted by ASpec Authors from NVIDIA cuDNN Frontend cb8cec8.

"""Correctness tests for ASpec's DSpark Q/K CuTeDSL kernel."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUTEDSL_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the ASpec CuTeDSL CUDA integration environment",
)

_EPSILON = 1.0e-6


def _reference(x, weight, positions, cos, sin):
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(-1, keepdim=True) + _EPSILON
    )
    normalized = normalized * weight.float()
    cos_rows = cos[positions].float().unsqueeze(1)
    sin_rows = sin[positions].float().unsqueeze(1)
    first, second = normalized.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (normalized * cos_rows + rotated * sin_rows).to(x.dtype)


def _inputs(batch: int, context_len: int, draft_len: int):
    generator = torch.Generator(device="cuda").manual_seed(
        batch * 10_000 + context_len * 100 + draft_len
    )
    dtype = torch.bfloat16
    h_q, h_kv, head_dim = 8, 2, 128
    query_tokens = batch * draft_len
    context_tokens = batch * context_len
    max_position = 67
    query = torch.randn(
        query_tokens,
        h_q,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    context_key = torch.randn(
        context_tokens,
        h_kv,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    draft_key = torch.randn(
        query_tokens,
        h_kv,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    query_weight = torch.randn(
        head_dim, generator=generator, device="cuda", dtype=dtype
    )
    key_weight = torch.randn(
        head_dim, generator=generator, device="cuda", dtype=dtype
    )
    angles = torch.randn(
        max_position,
        head_dim // 2,
        generator=generator,
        device="cuda",
    )
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    positions = tuple(
        torch.randint(
            max_position,
            (tokens,),
            generator=generator,
            device="cuda",
        )
        for tokens in (query_tokens, context_tokens, query_tokens)
    )
    return (
        query,
        context_key,
        draft_key,
        query_weight,
        key_weight,
        cos,
        sin,
        *positions,
    )


@pytest.mark.parametrize(
    ("batch", "context_len", "draft_len"),
    ((1, 5, 3), (1, 17, 7), (2, 5, 3)),
)
def test_forward_backward_matches_fp32_reference_and_is_deterministic(
    batch,
    context_len,
    draft_len,
):
    from aspec.kernels.cutedsl.dspark_qk_norm_rope import (
        compile_dspark_qk_norm_rope,
        run_dspark_qk_norm_rope,
    )
    from aspec.kernels.cutedsl.dspark_qk_norm_rope_bwd import (
        compile_dspark_qk_norm_rope_bwd,
        dspark_n_ctas_for,
        run_dspark_qk_norm_rope_bwd,
    )

    (
        query,
        context_key,
        draft_key,
        query_weight,
        key_weight,
        cos,
        sin,
        position_query,
        position_context,
        position_draft,
    ) = _inputs(batch, context_len, draft_len)
    query_tokens = batch * draft_len
    key_tokens = batch * (context_len + draft_len)
    h_q = query.shape[1]
    h_kv = context_key.shape[1]
    head_dim = query.shape[2]
    stream = torch.cuda.current_stream().cuda_stream

    forward = compile_dspark_qk_norm_rope(
        dtype=query.dtype,
        h_q=h_q,
        h_kv=h_kv,
        d=head_dim,
        rope_dim=head_dim,
        eps=_EPSILON,
        defer_secondary_loads=False,
        batch_one=batch == 1,
    )
    query_output = torch.empty_like(query)
    key_output = torch.empty(
        key_tokens, h_kv, head_dim, device="cuda", dtype=query.dtype
    )
    rstd_query = torch.empty(query_tokens, h_q, device="cuda", dtype=torch.float32)
    rstd_key = torch.empty(key_tokens, h_kv, device="cuda", dtype=torch.float32)
    run_dspark_qk_norm_rope(
        forward,
        query,
        context_key,
        draft_key,
        query_output,
        key_output,
        query_weight,
        key_weight,
        cos,
        sin,
        position_query,
        position_context,
        position_draft,
        rstd_query,
        rstd_key,
        batch=batch,
        context_len=context_len,
        draft_len=draft_len,
        stream=stream,
    )

    query_reference = _reference(
        query, query_weight, position_query, cos, sin
    )
    context_reference = _reference(
        context_key, key_weight, position_context, cos, sin
    ).view(batch, context_len, h_kv, head_dim)
    draft_reference = _reference(
        draft_key, key_weight, position_draft, cos, sin
    ).view(batch, draft_len, h_kv, head_dim)
    key_reference = torch.cat(
        (context_reference, draft_reference), dim=1
    ).view_as(key_output)
    torch.testing.assert_close(query_output, query_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key_output, key_reference, rtol=2e-2, atol=2e-2)

    grad_query = torch.randn_like(query_output)
    grad_key = torch.randn_like(key_output)
    backward = compile_dspark_qk_norm_rope_bwd(
        dtype=query.dtype,
        h_q=h_q,
        h_kv=h_kv,
        d=head_dim,
        rope_dim=head_dim,
        eps=_EPSILON,
        batch_one=batch == 1,
    )
    n_ctas_query, n_ctas_key = dspark_n_ctas_for(
        backward, query_tokens, key_tokens
    )

    def run_backward():
        grad_inputs = (
            torch.empty_like(query),
            torch.empty_like(context_key),
            torch.empty_like(draft_key),
        )
        partial_query = torch.empty(
            n_ctas_query, head_dim, device="cuda", dtype=torch.float32
        )
        partial_key = torch.empty(
            n_ctas_key, head_dim, device="cuda", dtype=torch.float32
        )
        grad_query_weight = torch.empty(head_dim, device="cuda", dtype=torch.float32)
        grad_key_weight = torch.empty_like(grad_query_weight)
        run_dspark_qk_norm_rope_bwd(
            backward,
            grad_query,
            grad_key,
            query,
            context_key,
            draft_key,
            rstd_query,
            rstd_key,
            query_weight,
            key_weight,
            cos,
            sin,
            position_query,
            position_context,
            position_draft,
            *grad_inputs,
            partial_query,
            partial_key,
            grad_query_weight,
            grad_key_weight,
            batch=batch,
            context_len=context_len,
            draft_len=draft_len,
            stream=stream,
        )
        return (*grad_inputs, grad_query_weight, grad_key_weight)

    actual = run_backward()
    repeated = run_backward()
    for first, second in zip(actual, repeated, strict=True):
        assert torch.equal(first, second)

    query_float = query.float().detach().requires_grad_(True)
    context_float = context_key.float().detach().requires_grad_(True)
    draft_float = draft_key.float().detach().requires_grad_(True)
    query_weight_float = query_weight.float().detach().requires_grad_(True)
    key_weight_float = key_weight.float().detach().requires_grad_(True)
    query_oracle = _reference(
        query_float, query_weight_float, position_query, cos, sin
    )
    context_oracle = _reference(
        context_float, key_weight_float, position_context, cos, sin
    ).view(batch, context_len, h_kv, head_dim)
    draft_oracle = _reference(
        draft_float, key_weight_float, position_draft, cos, sin
    ).view(batch, draft_len, h_kv, head_dim)
    key_oracle = torch.cat((context_oracle, draft_oracle), dim=1).view_as(key_output)
    torch.autograd.backward((query_oracle, key_oracle), (grad_query, grad_key))
    expected = (
        query_float.grad,
        context_float.grad,
        draft_float.grad,
        query_weight_float.grad,
        key_weight_float.grad,
    )
    tolerances = ((2e-2, 2e-2),) * 3 + ((1e-5, 2e-5),) * 2
    for result, reference, (rtol, atol) in zip(
        actual, expected, tolerances, strict=True
    ):
        torch.testing.assert_close(result.float(), reference, rtol=rtol, atol=atol)
