# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Internal/native GQA autograd checks, including ragged and empty rows."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires CUDA cuDNN GQA layout integration",
)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("include_lse", [False, True])
def test_internal_gqa_matches_native_output_lse_and_all_gradients(thd, include_lse):
    ratio = 4
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(924)
    batch, queries, keys, kv_heads, dim = 2, 43, 179, 2, 128
    heads = kv_heads * ratio
    data = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for shape in (
            (batch, queries, heads, dim),
            (batch, keys, kv_heads, dim),
            (batch, keys, kv_heads, dim),
        )
    ]
    lengths = [37, 43] if thd else [queries, queries]
    key_lengths = [163, 179] if thd else [keys, keys]
    if thd:
        data = [
            torch.cat([value[0, : lengths[0] if i == 0 else key_lengths[0]], value[1]])
            for i, value in enumerate(data)
        ]
    endpoints = []
    for qlen, klen in zip(lengths, key_lengths, strict=True):
        ends = torch.arange(qlen, device="cuda") % (klen - 9) + 1
        row = torch.stack(
            (torch.zeros_like(ends), torch.zeros_like(ends), ends, ends + 3, ends + 7)
        )
        row[:, 11] = 0  # Empty visibility must preserve -inf LSE and zero output.
        endpoints.append(row)
    mask = CudnnFlexMask(
        torch.cat(endpoints, dim=-1).int()[None].contiguous(),
        pack_gqa=False,
        **(
            dict(
                cu_seqlens_q=torch.tensor(
                    [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
                ),
                cu_seqlens_k=torch.tensor(
                    [0, key_lengths[0], sum(key_lengths)], device="cuda", dtype=torch.int32
                ),
                max_seqlen_q=max(lengths),
                max_seqlen_k=max(key_lengths),
            )
            if thd
            else {}
        ),
    )
    dout = torch.randn_like(data[0])  # Keep the actual attention dO boundary BF16.
    lse_shape = (heads, sum(lengths)) if thd else (batch, heads, queries)
    dlse = torch.randn(lse_shape, device="cuda", dtype=torch.float32) * 0.01
    results = []
    for mode in ("native", "internal"):
        q, k, v = [x.clone().requires_grad_() for x in data]
        output, lse = cudnn_flex_attention(
            q, k, v, mask=mask, scale=dim**-0.5, gqa_mode=mode, return_lse=True
        )
        assert output.shape == q.shape and lse.shape == lse_shape
        gradients = (
            torch.autograd.grad((output, lse), (q, k, v), (dout, dlse))
            if include_lse
            else torch.autograd.grad(output, (q, k, v), dout)
        )
        assert all(torch.isfinite(g).all() for g in gradients)
        results.append((output, lse, gradients))
    for actual, expected in zip(results[1][:2], results[0][:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.001)
    for actual, expected in zip(results[1][2], results[0][2], strict=True):
        relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp(
            min=1e-20
        )
        assert relative < 0.0005, relative.item()


def test_internal_gqa_multiple_live_autograd_graphs():
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(330)
    qshape, kshape = (1, 71, 8, 128), (1, 229, 2, 128)
    ends = torch.full((71,), 229, device="cuda", dtype=torch.int32)
    zeros = torch.zeros_like(ends)
    mask = CudnnFlexMask(torch.stack((zeros, zeros, ends, ends, ends))[None])
    inputs = [
        [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for shape in (qshape, kshape, kshape)
        ]
        for _ in range(2)
    ]
    dout = [torch.randn(qshape, device="cuda", dtype=torch.bfloat16) for _ in inputs]
    outputs = [
        cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode="internal")
        for values in inputs
    ]
    # Reuse the immutable plan with distinct pending graphs, then consume in reverse order.
    actual = [None, None]
    for index in (1, 0):
        actual[index] = torch.autograd.grad(outputs[index], inputs[index], dout[index])
    for values, gradient, expected_dout in zip(inputs, actual, dout, strict=True):
        output = cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode="native")
        expected = torch.autograd.grad(output, values, expected_dout)
        for got, ref in zip(gradient, expected, strict=True):
            error = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20)
            assert error < 0.0005


def test_internal_gqa_reuses_jit_when_anchor_metadata_cardinality_changes():
    from aspec.models.ops import cudnn_gqa_backward as joint
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(195)
    shapes = ((1, 133, 8, 128), (1, 4097, 2, 128), (1, 4097, 2, 128))
    values = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_() for shape in shapes
    ]
    dout = torch.randn_like(values[0])
    joint._COMPILED.clear()
    first_cache_keys = None
    metadata_sizes = []
    for endpoint in (997, 1523, 2207):
        ends = torch.full((133,), endpoint, device="cuda", dtype=torch.int32)
        zeros = torch.zeros_like(ends)
        mask = CudnnFlexMask(torch.stack((zeros, zeros, ends, ends, ends))[None])
        grads = []
        for mode in ("internal", "native"):
            output = cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode=mode)
            grads.append(torch.autograd.grad(output, values, dout))
        plan = next(iter(mask._joint_backward_plans.values()))
        metadata_sizes.append(tuple(None if t is None else tuple(t.shape) for t in plan.sparse))
        for actual, reference in zip(*grads, strict=True):
            relative = (actual.float() - reference.float()).norm() / reference.float().norm()
            assert relative < 0.0005
        keys = tuple(joint._COMPILED)
        if first_cache_keys is None:
            first_cache_keys = keys
        else:
            assert keys == first_cache_keys, "Changing sparse counts must not recompile"
    assert len(set(metadata_sizes)) == 3


@pytest.mark.parametrize("block", [5, 7])
def test_internal_gqa_dspark_block_against_fp64_math(block):
    """Both routes must retain native BF16 accuracy against an independent oracle."""
    import json

    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(729)
    context, queries, kv_heads, dim = 55, 4 * block, 2, 128
    data = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
        for shape in (
            (1, queries, kv_heads * 4, dim),
            (1, context + queries, kv_heads, dim),
            (1, context + queries, kv_heads, dim),
        )
    ]
    anchor = torch.tensor([3, 11, 27, 49], device="cuda").repeat_interleave(block)
    draft_start = context + torch.arange(queries, device="cuda") // block * block
    draft_end = draft_start + block
    mask = CudnnFlexMask(
        torch.stack((anchor + 1, draft_start, draft_end, draft_end, draft_end))
        .int()[None]
        .contiguous(),
        pack_gqa=False,
    )
    k_index = torch.arange(context + queries, device="cuda")[None]
    visible = (k_index <= anchor[:, None]) | (
        (k_index >= draft_start[:, None]) & (k_index < draft_end[:, None])
    )
    dqout = torch.randn_like(data[0])
    q, k, v = [t.double().requires_grad_() for t in data]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(4, dim=2)) * dim**-0.5
    probabilities = scores.masked_fill(~visible[None, None], -torch.inf).softmax(-1)
    reference = torch.einsum("bhqk,bkhd->bqhd", probabilities, v.repeat_interleave(4, dim=2))
    reference_grads = torch.autograd.grad(reference, (q, k, v), dqout.double())
    errors = {}
    for mode in ("native", "internal"):
        values = [t.clone().requires_grad_() for t in data]
        actual = cudnn_flex_attention(*values, mask=mask, scale=dim**-0.5, gqa_mode=mode)
        gradients = torch.autograd.grad(actual, values, dqout)
        errors[mode] = [
            float(((a.double() - b).norm() / b.norm()).detach())
            for a, b in zip((actual, *gradients), (reference, *reference_grads), strict=True)
        ]
    print("FP64_ORACLE_ERRORS", json.dumps(errors), flush=True)
    for candidate, native in zip(errors["internal"], errors["native"], strict=True):
        assert candidate < 0.01
        assert candidate <= native * 1.02 + 1e-6


@pytest.mark.parametrize("lse_only", [False, True])
def test_internal_gqa_nondefault_stream_and_lse_only(lse_only):
    """TVM-FFI launches must use the caller's stream, including LSE-only backward."""
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(191)
        values = [
            torch.randn(s, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for s in ((1, 137, 8, 128), (1, 259, 2, 128), (1, 259, 2, 128))
        ]
        ends = torch.full((137,), 259, device="cuda", dtype=torch.int32)
        zeros = torch.zeros_like(ends)
        mask = CudnnFlexMask(torch.stack((zeros, zeros, ends, ends, ends))[None])
        results = []
        for mode in ("native", "internal"):
            output, lse = cudnn_flex_attention(
                *values, mask=mask, scale=128**-0.5, gqa_mode=mode, return_lse=True
            )
            loss = lse.sum() if lse_only else output.float().square().mean() + lse.mean()
            if lse_only and mode == "native":
                # FE 1.31 rejects the stride-zero dLSE generated by sum().
                # Supply the same all-ones gradient in its required layout.
                gradients = torch.autograd.grad(lse, values, torch.ones_like(lse))
            else:
                gradients = torch.autograd.grad(loss, values)
            results.append(gradients)
        for actual, expected in zip(results[1], results[0], strict=True):
            # LSE has no dependence on V. Native autograd may return a zero tensor.
            assert torch.isfinite(actual).all()
            relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
            assert relative < 0.0005
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("case", ["fp16", "ratio", "noncontiguous", "head_mask", "empty"])
def test_internal_gqa_rejects_unsupported_requests(case):
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    q = torch.randn(1, 17, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 31, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ends = torch.full((17,), 31, device="cuda", dtype=torch.int32)
    zeros = torch.zeros_like(ends)
    mask = CudnnFlexMask(torch.stack((zeros, zeros, ends, ends, ends))[None])
    if case == "fp16":
        q, k, v = q.half(), k.half(), v.half()
    elif case == "ratio":
        q = q[:, :, :4].contiguous()
    elif case == "noncontiguous":
        q = torch.empty(1, 34, 8, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif case == "head_mask":
        mask.mask_func = mask.mask_func.expand(8, -1, -1).contiguous()
    else:
        q = q[:, :0]
    with pytest.raises(ValueError, match="internal GQA backward requires"):
        cudnn_flex_attention(q, k, v, mask=mask, scale=128**-0.5, gqa_mode="internal")


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("block", [5, 7])
def test_internal_gqa_relay_replay_with_changing_payload(thd, block):
    """Exercise many dS relay phases, ragged tails, and refreshed graph inputs."""
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    torch.manual_seed(20261031)
    qlengths = [37 * block, 257 * block] if thd else [257 * block] * 2
    contexts = [1543, 4097] if thd else [4097] * 2
    klengths = [c + q for c, q in zip(contexts, qlengths, strict=True)]
    endpoints = []
    for context, qlength in zip(contexts, qlengths, strict=True):
        anchors = torch.linspace(1, context - block - 1, qlength // block, device="cuda")
        prefix = anchors.long().repeat_interleave(block)
        start = context + torch.arange(qlength, device="cuda") // block * block
        zeros = torch.zeros_like(start)
        row = torch.stack((zeros, zeros, prefix, start, start + block))
        row[:, 11] = 0
        endpoints.append(row)
    mask = CudnnFlexMask(
        torch.cat(endpoints, dim=-1).int()[None].contiguous(),
        **(
            dict(
                cu_seqlens_q=torch.tensor(
                    [0, qlengths[0], sum(qlengths)], dtype=torch.int32, device="cuda"
                ),
                cu_seqlens_k=torch.tensor(
                    [0, klengths[0], sum(klengths)], dtype=torch.int32, device="cuda"
                ),
                max_seqlen_q=max(qlengths),
                max_seqlen_k=max(klengths),
            )
            if thd else {}
        ),
    )
    qshape = (sum(qlengths), 8, 128) if thd else (2, qlengths[0], 8, 128)
    kshape = (sum(klengths), 2, 128) if thd else (2, klengths[0], 2, 128)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for shape in (qshape, kshape, kshape)
    ]
    dout = torch.randn_like(inputs[0])

    def gradients(mode):
        output = cudnn_flex_attention(
            *inputs, mask=mask, scale=128**-0.5, gqa_mode=mode
        )
        return output, torch.autograd.grad(output, inputs, dout)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gradients("internal")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual_output, actual_gradients = gradients("internal")
    torch.cuda.current_stream().wait_stream(stream)

    with torch.cuda.stream(stream):
        for _ in range(8):
            with torch.no_grad():
                for tensor in (*inputs, dout):
                    tensor.normal_()
            expected_output, expected_gradients = gradients("native")
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
                assert bool(torch.isfinite(actual).all())
                relative = (actual.float() - expected.float()).norm() / expected.float().norm()
                assert relative < 5e-4, relative.item()
