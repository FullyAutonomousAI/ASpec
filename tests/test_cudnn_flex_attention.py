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

import os
from dataclasses import replace

import pytest
import torch

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.draft.dflash import (
    DFlashAttention,
    DFlashConfig,
    DFlashRMSNorm,
    _dflash_rms_norm,
)
from aspec.models.draft.dspark import K3DSparkConfig, K3DSparkMLAAttention
from aspec.models.ops.cudnn_flex_attention import (
    CudnnFlexMask,
    _pack_physical_gqa_query,
    _prefer_thd_physical_gqa,
    _unpack_physical_gqa_lse,
    _unpack_physical_gqa_output,
    build_dspark_mla_mask,
    cudnn_flex_attention,
)
from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    _build_cudnn_thd_mask,
    _visibility,
    direct_cp_attention,
    direct_cp_attention_observation,
)
from aspec.models.ops.flex_attention import compile_friendly_create_block_mask


def _endpoint_visible(mask_func, batch, query, key, query_length):
    flat_query = batch * query_length + query
    endpoints = mask_func[0, :, flat_query]
    visible = key < int(endpoints[0])
    for index in range(1, endpoints.numel(), 2):
        visible |= int(endpoints[index]) <= key < int(endpoints[index + 1])
    return visible


def test_cudnn_mask_supports_pytorch_dataclass_replacement():
    mask = build_dspark_mla_mask(
        anchor_positions=torch.tensor([[1, 4]]),
        block_keep_mask=torch.tensor([[True, True]]),
        context_length=8,
        block_size=2,
    )
    replaced = replace(mask, mask_func=mask.mask_func.clone())
    assert replaced._plans is mask._plans
    query = torch.empty(1, 4, 2, 128)
    key = torch.empty(1, 8, 1, 128)
    value = torch.empty(1, 8, 1, 128)
    assert mask._plan_key(query, key, value, True) != replaced._plan_key(
        query, key, value, True
    )


def test_direct_cp_observation_survives_dataclass_replacement():
    metadata = DirectCPMetadata(
        anchor_positions=torch.tensor([[1, 3]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        context_position_ids=torch.arange(8).view(1, -1),
        block_size=2,
        cp_rank=0,
        cp_size=1,
        group=None,
        backend="cudnn_flex_attention",
    )
    copied = replace(metadata, attention_observation={})

    direct_cp_attention_observation(copied)["kernel_backend"] = (
        "cudnn_flex_attention"
    )

    assert direct_cp_attention_observation(metadata) == {
        "kernel_backend": "cudnn_flex_attention"
    }


@pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the CUDA pointwise-kernel integration environment",
)
def test_dflash_quack_rmsnorm_matches_fp32_reference_math():
    generator = torch.Generator(device="cuda").manual_seed(731)
    hidden = torch.randn(
        3, 17, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)
    gradient = torch.randn(
        hidden.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    module = DFlashRMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    module.weight.data.copy_(weight)
    actual_input = hidden.detach().clone().requires_grad_(True)
    actual = module(actual_input)
    actual_grads = torch.autograd.grad(actual, (actual_input, module.weight), gradient)

    reference_input = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = _dflash_rms_norm(reference_input, reference_weight, 1e-6)
    reference_grads = torch.autograd.grad(
        reference, (reference_input, reference_weight), gradient
    )

    torch.testing.assert_close(actual, reference, atol=0.015, rtol=0.02)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual_grad, reference_grad, atol=0.03, rtol=0.04)


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_physical_gqa_layout_round_trips_query_heads(thd, ratio):
    heads = 2 * ratio
    query = torch.arange(2 * 3 * heads * 4).reshape(2, 3, heads, 4)
    if thd:
        query = query.flatten(0, 1)
    physical = _pack_physical_gqa_query(query, kv_heads=2)

    assert physical.shape == ((6 * ratio, 2, 4) if thd else (2, 3 * ratio, 2, 4))
    restored = _unpack_physical_gqa_output(
        physical,
        query_length=6 if thd else 3,
        query_heads=heads,
    )
    torch.testing.assert_close(restored, query)


@pytest.mark.parametrize("thd", [False, True])
def test_physical_gqa_lse_restores_head_and_query_order(thd):
    logical = torch.arange(2 * 8 * 7).reshape(2, 8, 7)
    if thd:
        logical = logical[0]
    prefix = () if thd else (2,)
    physical = logical.reshape(*prefix, 2, 4, 7).transpose(-1, -2).reshape(*prefix, 2, 28)
    restored = _unpack_physical_gqa_lse(physical, query_length=7, query_heads=8)
    assert torch.equal(restored, logical)


def test_thd_physical_gqa_reuses_metadata_and_invalidates_changed_lengths():
    mask = CudnnFlexMask(mask_func=torch.zeros(1, 5, 5, dtype=torch.int32),
                         cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
                         cu_seqlens_k=torch.tensor([0, 7, 16], dtype=torch.int32),
                         max_seqlen_q=3, max_seqlen_k=9)
    physical = mask.for_physical_gqa(3)
    assert physical is mask.for_physical_gqa(3)
    assert physical.cu_seqlens_q.tolist() == [0, 6, 15]
    assert physical.max_seqlen_q == 9
    assert physical.cu_seqlens_k is mask.cu_seqlens_k
    assert physical.max_seqlen_k == 9
    query, key = torch.empty(5, 6, 128), torch.empty(16, 2, 128)
    plan_key = mask._plan_key(query, key, key, None)
    mask.cu_seqlens_q[1] = 3
    changed = mask.for_physical_gqa(3)
    assert changed is not physical
    assert changed.cu_seqlens_q.tolist() == [0, 9, 15]
    assert mask._plan_key(query, key, key, None) != plan_key


@pytest.mark.parametrize(("queries", "keys", "heads", "kv_heads", "expected"), [
    (3584, 7680, 32, 8, False), (16121, 135231, 32, 8, True),
    (10542, 137062, 32, 8, True), (700, 2799, 32, 8, False),
    (700, 2800, 32, 8, True), (700, 2800, 32, 32, False),
    (700, 2100, 6, 2, True), (700, 2100, 8, 3, False),
])
def test_thd_gqa_policy_uses_local_geometry(queries, keys, heads, kv_heads, expected):
    assert _prefer_thd_physical_gqa(queries, keys, heads, kv_heads) is expected


def test_physical_gqa_mask_repeats_queries_and_reuses_derived_mask():
    mask = build_dspark_mla_mask(
        anchor_positions=torch.tensor([[1, 4]]),
        block_keep_mask=torch.tensor([[True, True]]),
        context_length=8,
        block_size=2,
    )

    physical = mask.for_physical_gqa(4)

    assert physical is mask.for_physical_gqa(4)
    assert physical.pack_gqa is False
    torch.testing.assert_close(
        physical.mask_func,
        mask.mask_func.repeat_interleave(4, dim=-1),
    )


@pytest.mark.parametrize("block_size", (5, 7))
@pytest.mark.parametrize(
    ("is_causal", "sliding_window"),
    ((False, None), (True, None), (False, 4), (True, 4)),
)
def test_cudnn_dspark_endpoints_match_mask_mod(block_size, is_causal, sliding_window):
    context_length = 20
    anchors = torch.tensor([[3, 12], [7, 16]])
    keep = torch.tensor([[True, False], [True, True]])
    result = build_dspark_mla_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_length=context_length,
        block_size=block_size,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    assert result.mask_func.shape[1] == (3 if sliding_window is None else 5)
    mask_mod = _create_dflash_mask_mod(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=context_length,
        block_size=block_size,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    query_length = anchors.shape[1] * block_size
    key_length = context_length + query_length
    for batch in range(anchors.shape[0]):
        for query in range(query_length):
            for key in range(key_length):
                expected = bool(mask_mod(batch, 0, query, key))
                actual = _endpoint_visible(
                    result.mask_func, batch, query, key, query_length
                )
                assert actual == expected


@pytest.mark.parametrize("block_size", (5, 7))
def test_cudnn_dspark_endpoints_match_packed_document_mask(block_size):
    context_length = 20
    anchors = torch.tensor([[3, 8, 16]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]])
    result = build_dspark_mla_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_length=context_length,
        block_size=block_size,
        ctx_doc_ids=doc_ids,
    )
    mask_mod = _create_dflash_mask_mod(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=context_length,
        block_size=block_size,
        ctx_doc_ids=doc_ids,
    )
    query_length = anchors.shape[1] * block_size
    key_length = context_length + query_length
    for query in range(query_length):
        for key in range(key_length):
            assert _endpoint_visible(result.mask_func, 0, query, key, query_length) == bool(
                mask_mod(0, 0, query, torch.tensor(key))
            )


def test_cudnn_dspark_endpoints_reject_invalid_geometry():
    anchors = torch.tensor([[1, 2]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    with pytest.raises(ValueError, match="inside the context"):
        build_dspark_mla_mask(
            anchor_positions=torch.tensor([[1, 8]]),
            block_keep_mask=keep,
            context_length=8,
            block_size=5,
        )
    with pytest.raises(ValueError, match="sliding_window"):
        build_dspark_mla_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_length=8,
            block_size=5,
            sliding_window=0,
        )
    with pytest.raises(ValueError, match="contiguous run"):
        build_dspark_mla_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_length=8,
            block_size=5,
            ctx_doc_ids=torch.tensor([[0, 0, 1, 1, 0, 0, 2, 2]]),
        )


@pytest.mark.parametrize("cp_size", (1, 2, 4))
@pytest.mark.parametrize(
    ("context_window", "query_window", "draft_is_causal"),
    ((None, None, False), (6, None, False), (None, 4, True)),
)
def test_cudnn_thd_direct_cp_endpoints_match_reference_visibility(
    cp_size, context_window, query_window, draft_is_causal
):
    context_length = 24
    block_size = 3
    anchors = torch.tensor([[1, 3, 4, 5, 9, 11, 17, 19]])
    keep = torch.tensor([[True, True, False, True, True, True, True, False]])
    global_docs = torch.arange(3).repeat_interleave(8).view(1, -1)

    for cp_rank in range(cp_size):
        per_document = []
        for doc_start in range(0, context_length, 8):
            shard_start = doc_start + (8 * cp_rank // cp_size)
            shard_end = doc_start + (8 * (cp_rank + 1) // cp_size)
            per_document.append(torch.arange(shard_start, shard_end))
        positions = torch.cat(per_document).view(1, -1)
        context_docs = global_docs.gather(1, positions)
        metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=positions,
            context_doc_ids=context_docs,
            anchor_doc_ids=global_docs.gather(1, anchors),
            block_size=block_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
            group=None if cp_size == 1 else object(),
            sequence_length=context_length,
            backend="cudnn_flex_attention",
            context_window=context_window,
            query_window=query_window,
            draft_is_causal=draft_is_causal,
        )
        query_length = anchors.shape[1] * block_size
        key_length = metadata.context_length + metadata.queries_per_rank
        mask = _build_cudnn_thd_mask(
            metadata,
            device=torch.device("cpu"),
            query_length=query_length,
            key_length=key_length,
        )
        expected = _visibility(metadata, torch.device("cpu"))
        actual = torch.zeros_like(expected)
        for query in range(query_length):
            for key in range(key_length):
                actual[0, query, key] = _endpoint_visible(
                    mask.mask_func, 0, query, key, query_length
                )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            mask.cu_seqlens_q,
            torch.tensor([0, query_length], dtype=torch.int32),
        )
        torch.testing.assert_close(
            mask.cu_seqlens_k,
            torch.tensor([0, key_length], dtype=torch.int32),
        )


def test_dflash_gqa_cudnn_backend_fails_closed_without_mask():
    config = DFlashConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        num_target_layers=1,
        target_hidden_size=512,
        target_num_hidden_layers=1,
    )
    attention = DFlashAttention(config)
    attention.set_attention_backend("cudnn_flex_attention")
    draft = torch.randn(1, 8, 512)
    context = torch.randn(1, 16, 512)
    with pytest.raises(RuntimeError, match="requires a CUDA cuDNN mask"):
        attention(
            draft,
            context,
            torch.arange(8).unsqueeze(0),
            torch.arange(16).unsqueeze(0),
        )


@pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the cuDNN Frontend FlexAttention CUDA integration environment",
)
def test_cudnn_thd_cp1_forward_lse_and_backward_match_math():
    pytest.importorskip("cudnn.flex_attention")
    generator = torch.Generator(device="cuda").manual_seed(821)
    batch, query_heads, kv_heads = 2, 4, 2
    context_length, block_size = 128, 5
    anchors = torch.tensor([[17, 49, 81, 113], [19, 51, 83, 115]], device="cuda")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    query_length = anchors.shape[1] * block_size
    docs = torch.arange(4, device="cuda").repeat_interleave(32).view(1, -1)
    docs = docs.expand(batch, -1)
    shapes = (
        (batch, query_heads, query_length, 128),
        (batch, kv_heads, context_length, 128),
        (batch, kv_heads, context_length, 128),
        (batch, kv_heads, query_length, 128),
        (batch, kv_heads, query_length, 128),
    )
    inputs = [
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for shape in shapes
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    common = dict(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=torch.arange(context_length, device="cuda").view(1, -1).expand(batch, -1),
        context_doc_ids=docs,
        anchor_doc_ids=docs.gather(1, anchors),
        block_size=block_size,
        cp_rank=0,
        cp_size=1,
        group=None,
        sequence_length=context_length,
    )
    actual, actual_lse = direct_cp_attention(
        *inputs, DirectCPMetadata(**common, backend="cudnn_flex_attention")
    )
    expected, expected_lse = direct_cp_attention(
        *reference_inputs, DirectCPMetadata(**common, backend="math")
    )
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.035)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.025, rtol=0.035)

    dout = torch.randn(actual.shape, device="cuda", dtype=torch.float32, generator=generator)
    dlse = torch.randn(actual_lse.shape, device="cuda", dtype=torch.float32, generator=generator)
    actual_loss = (actual.float() * dout).sum() + 0.01 * (actual_lse * dlse).sum()
    expected_loss = (expected.float() * dout).sum() + 0.01 * (expected_lse * dlse).sum()
    actual_grads = torch.autograd.grad(actual_loss, inputs)
    expected_grads = torch.autograd.grad(expected_loss, reference_inputs)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, reference, atol=0.06, rtol=0.08)


@pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the cuDNN Frontend FlexAttention CUDA integration environment",
)
@pytest.mark.parametrize("block_size", (5, 7))
def test_cudnn_dspark_mla_forward_backward_matches_dense(block_size):
    pytest.importorskip("cudnn.flex_attention")
    context_length, num_anchors, num_heads = 256, 4, 4
    anchors = torch.tensor([[17, 81, 145, 223]], device="cuda")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    mask = build_dspark_mla_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_length=context_length,
        block_size=block_size,
    )
    query_length = num_anchors * block_size
    key_length = context_length + query_length
    generator = torch.Generator(device="cuda").manual_seed(block_size)
    query = torch.randn(
        1,
        query_length,
        num_heads,
        192,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    key = torch.randn(
        1,
        key_length,
        num_heads,
        192,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    # Match kv_b_proj followed by split: V is BSHD but has a 320-element
    # head stride rather than being separately contiguous.
    kv = torch.randn(
        1,
        key_length,
        num_heads,
        320,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    value = kv[..., 192:]
    assert not value.is_contiguous()
    dout = torch.randn(
        1,
        query_length,
        num_heads,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    output = cudnn_flex_attention(
        query,
        key,
        value,
        mask=mask,
        scale=192**-0.5,
    )
    actual_grads = torch.autograd.grad(output, (query, key, value), dout)

    reference_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)
    ]
    ref_query, ref_key, ref_value = reference_inputs
    scores = torch.einsum("bqhd,bkhd->bhqk", ref_query.float(), ref_key.float())
    scores = scores * (192**-0.5)
    keys = torch.arange(key_length, device="cuda").view(1, 1, key_length)
    nfunc = mask.mask_func.shape[1]
    endpoints = mask.mask_func[0].view(nfunc, 1, query_length).permute(1, 2, 0)
    visible = keys < endpoints[..., 0:1]
    for index in range(1, nfunc, 2):
        visible |= (endpoints[..., index : index + 1] <= keys) & (
            keys < endpoints[..., index + 1 : index + 2]
        )
    probabilities = torch.softmax(
        scores.masked_fill(~visible.unsqueeze(1), float("-inf")), dim=-1
    )
    reference = torch.einsum(
        "bhqk,bkhd->bqhd", probabilities, ref_value.float()
    ).to(torch.bfloat16)
    reference_grads = torch.autograd.grad(reference, reference_inputs, dout)

    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.02)
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)


@pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the cuDNN Frontend FlexAttention CUDA integration environment",
)
@pytest.mark.parametrize("block_size", (5, 7))
@pytest.mark.parametrize("packed", (False, True))
def test_dflash_gqa_cudnn_backend_matches_aspec_flex_attention(block_size, packed):
    pytest.importorskip("cudnn.flex_attention")
    config = DFlashConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        num_target_layers=1,
        target_hidden_size=512,
        target_num_hidden_layers=1,
    )
    generator = torch.Generator(device="cuda").manual_seed(200 + block_size)
    attention = DFlashAttention(config).to(device="cuda", dtype=torch.bfloat16)
    context_length, num_anchors = (253 if packed else 256), 4
    draft_length = block_size * num_anchors
    anchors = torch.tensor([[17, 81, 145, 223]], device="cuda")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    draft_position_ids = anchors.repeat_interleave(block_size, dim=1)
    context_position_ids = torch.arange(context_length, device="cuda").unsqueeze(0)
    doc_ids = None
    if packed:
        doc_ids = torch.repeat_interleave(
            torch.arange(3, device="cuda"),
            torch.tensor([96, 96, context_length - 192], device="cuda"),
        )[None]
        starts = torch.tensor([0, 96, 192], device="cuda")
        context_position_ids = context_position_ids - starts[doc_ids]
        draft_position_ids = torch.gather(context_position_ids, 1, anchors).repeat_interleave(block_size, dim=1)

    mask_mod = _create_dflash_mask_mod(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=context_length,
        block_size=block_size,
        ctx_doc_ids=doc_ids,
    )
    flex_mask = compile_friendly_create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=draft_length,
        KV_LEN=context_length + draft_length,
        device=torch.device("cuda"),
    )
    cudnn_mask = build_dspark_mla_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_length=context_length,
        block_size=block_size,
        ctx_doc_ids=doc_ids,
    )
    draft = torch.randn(
        1, draft_length, config.hidden_size,
        generator=generator, device="cuda", dtype=torch.bfloat16,
    )
    context = torch.randn(
        1, context_length, config.hidden_size,
        generator=generator, device="cuda", dtype=torch.bfloat16,
    )
    dout = torch.randn_like(draft)
    parameters = tuple(attention.parameters())

    attention.set_attention_backend("flex_attention")
    flex_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (draft, context))
    flex_output = attention(
        *flex_inputs, draft_position_ids, context_position_ids, block_mask=flex_mask
    )
    flex_grads = torch.autograd.grad(flex_output, (*flex_inputs, *parameters), dout)

    attention.set_attention_backend("cudnn_flex_attention")
    cudnn_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (draft, context))
    cudnn_output = attention(
        *cudnn_inputs, draft_position_ids, context_position_ids, block_mask=cudnn_mask
    )
    from aspec.models.ops.dspark_qk_norm_rope import (
        dspark_qk_norm_rope_available,
    )

    expected_qk_backend = (
        "aspec_cutedsl" if dspark_qk_norm_rope_available() else "eager"
    )
    assert attention.qk_prepare_backend == expected_qk_backend
    cudnn_grads = torch.autograd.grad(cudnn_output, (*cudnn_inputs, *parameters), dout)

    torch.testing.assert_close(cudnn_output, flex_output, atol=0.02, rtol=0.03)
    for actual, expected in zip(cudnn_grads, flex_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.05)


@pytest.mark.skipif(
    not (
        os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") == "1"
        and torch.cuda.is_available()
    ),
    reason="requires the cuDNN Frontend FlexAttention CUDA integration environment",
)
@pytest.mark.parametrize("block_size", (5, 7))
def test_k3_attention_cudnn_backend_matches_aspec_flex_attention(block_size):
    pytest.importorskip("cudnn.flex_attention")
    config = K3DSparkConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vocab_size=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        num_target_layers=1,
        target_hidden_size=256,
        target_num_hidden_layers=1,
        mask_token_id=511,
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
    )
    generator = torch.Generator(device="cuda").manual_seed(100 + block_size)
    attention = K3DSparkMLAAttention(config).to(device="cuda", dtype=torch.bfloat16)
    context_length, num_anchors = 256, 4
    draft_length = block_size * num_anchors
    anchors = torch.tensor([[17, 81, 145, 223]], device="cuda")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    draft_position_ids = anchors.repeat_interleave(block_size, dim=1)
    context_position_ids = torch.arange(context_length, device="cuda").unsqueeze(0)

    mask_mod = _create_dflash_mask_mod(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=context_length,
        block_size=block_size,
    )
    flex_mask = compile_friendly_create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=draft_length,
        KV_LEN=context_length + draft_length,
        device=torch.device("cuda"),
    )
    cudnn_mask = build_dspark_mla_mask(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_length=context_length,
        block_size=block_size,
    )

    draft = torch.randn(
        1,
        draft_length,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    context = torch.randn(
        1,
        context_length,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dout = torch.randn_like(draft)
    parameters = tuple(attention.parameters())

    attention.set_attention_backend("flex_attention")
    flex_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (draft, context))
    flex_output = attention(
        *flex_inputs,
        draft_position_ids,
        context_position_ids,
        block_mask=flex_mask,
    )
    flex_grads = torch.autograd.grad(
        flex_output, (*flex_inputs, *parameters), dout
    )

    attention.set_attention_backend("cudnn_flex_attention")
    cudnn_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (draft, context))
    cudnn_output = attention(
        *cudnn_inputs,
        draft_position_ids,
        context_position_ids,
        block_mask=cudnn_mask,
    )
    cudnn_grads = torch.autograd.grad(
        cudnn_output, (*cudnn_inputs, *parameters), dout
    )

    torch.testing.assert_close(cudnn_output, flex_output, atol=0.02, rtol=0.03)
    for actual, expected in zip(cudnn_grads, flex_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.05)
