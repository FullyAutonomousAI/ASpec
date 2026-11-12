# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit GQA execution routes must retain the default and fail closed."""

import pytest

from aspec.models.draft.dflash import DFlashConfig
from aspec.models.ops.cudnn_flex_attention import resolve_cudnn_gqa_mode


@pytest.mark.parametrize("physical", [False, True])
def test_default_preserves_existing_gqa_selection(physical):
    assert resolve_cudnn_gqa_mode("auto", physical_gqa=physical) == (
        "external" if physical else "native"
    )


@pytest.mark.parametrize("mode", ["native", "external", "internal"])
@pytest.mark.parametrize("physical", [False, True])
def test_explicit_gqa_mode_overrides_legacy_selection(mode, physical):
    assert resolve_cudnn_gqa_mode(mode, physical_gqa=physical) == mode
    assert DFlashConfig(cudnn_gqa_mode=mode).cudnn_gqa_mode == mode


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="GQA"):
        DFlashConfig(cudnn_gqa_mode="typo")
    with pytest.raises(ValueError, match="GQA"):
        resolve_cudnn_gqa_mode("typo")


@pytest.mark.parametrize("mode", ["native", "external", "internal"])
def test_explicit_mode_requires_cudnn_backend(mode):
    from types import SimpleNamespace

    from aspec.models.dflash import DFlashModel

    with pytest.raises(ValueError, match="requires attention_backend"):
        DFlashModel(
            SimpleNamespace(config=SimpleNamespace(cudnn_gqa_mode=mode)),
            attention_backend="flex_attention",
        )


def test_internal_mode_rejects_multi_rank_context_parallelism():
    from types import SimpleNamespace

    from aspec.models.dflash import DFlashModel

    with pytest.raises(NotImplementedError, match="CP=1"):
        DFlashModel(
            SimpleNamespace(config=SimpleNamespace(cudnn_gqa_mode="internal")),
            attention_backend="cudnn_flex_attention",
            draft_context_parallel_size=2,
        )


def test_pinned_frontend_fails_closed(monkeypatch):
    import aspec.models.ops.cudnn_gqa_backward as joint

    joint._validate_version.cache_clear()
    monkeypatch.setattr(joint, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="pinned"):
        joint._validate_version()
    joint._validate_version.cache_clear()


@pytest.mark.parametrize("mode", ["native", "external", "internal"])
@pytest.mark.parametrize("backend", ["cudnn_flex_attention", "math"])
def test_direct_cp_propagates_explicit_route_and_observation(monkeypatch, mode, backend):
    import torch

    import aspec.models.ops.cudnn_flex_attention as adapter
    import aspec.models.ops.direct_cp as cp

    seen = []

    def attention(q, k, v, **kwargs):
        seen.append(kwargs["gqa_mode"])
        return q * 2, torch.zeros(q.shape[1], q.shape[0])

    monkeypatch.setattr(adapter, "cudnn_flex_attention", attention)
    monkeypatch.setattr(cp, "_build_cudnn_thd_mask", lambda *a, **kw: object())
    source = torch.randn(1, 14, 8, 128, requires_grad=True)
    key = torch.randn(1, 2, 46, 128)
    metadata = cp.DirectCPMetadata(
        anchor_positions=torch.tensor([[7, 23]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        context_position_ids=torch.arange(32)[None],
        block_size=7, cp_rank=0, cp_size=1, group=None,
        sequence_length=32, backend=backend,
    )
    if backend == "math":
        with pytest.raises(ValueError, match="requires the cuDNN backend"):
            cp.direct_cp_attention_from_kv(
                source.transpose(1, 2), key, key, metadata, cudnn_gqa_mode=mode
            )
        assert not seen
        return
    out, _ = cp.direct_cp_attention_from_kv(
        source.transpose(1, 2), key, key, metadata, 128**-0.5,
        cudnn_gqa_mode=mode,
    )
    assert seen == [mode]
    observation = metadata.attention_observation
    assert observation["gqa_layout"] == {
        "native": "native", "external": "query_head_fold", "internal": "internal_joint"
    }[mode]
    assert observation["physical_query_tokens"] == (56 if mode == "external" else 14)
    out.sum().backward()
    assert observation["backward_observed"]


@pytest.mark.parametrize("mode", ["native", "external", "internal"])
@pytest.mark.parametrize("identity", [{"model_type": "k3_dspark"}, {"attention_mode": "mla"}])
def test_explicit_gqa_mode_cannot_be_ignored_by_mla(mode, identity):
    from types import SimpleNamespace

    from aspec.models.dflash import DFlashModel

    with pytest.raises(NotImplementedError, match="not supported for MLA"):
        DFlashModel(
            SimpleNamespace(config=SimpleNamespace(cudnn_gqa_mode=mode, **identity)),
            attention_backend="cudnn_flex_attention",
        )
