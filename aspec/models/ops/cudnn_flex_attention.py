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

"""cuDNN Frontend FlexAttention adapter for DSpark GQA and materialized MLA.

cuDNN Frontend consumes fixed BSHD or flattened variable-length THD Q/K/V
tensors and an interval-union mask. K3 DSpark materializes MLA K/V before
attention, so its Dqk=192, Dv=128 tensors fit that contract even though native
compressed-latent MLA is outside the FlexAttention API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

_SUPPORTED_HEAD_DIMS = {(128, 128), (192, 128)}


def _document_run_starts(ctx_doc_ids: torch.Tensor) -> torch.Tensor:
    """Return the start of the contiguous document run containing each token."""

    if ctx_doc_ids.ndim != 2:
        raise ValueError("ctx_doc_ids must have shape [batch, context_length]")
    positions = torch.arange(
        ctx_doc_ids.shape[1], device=ctx_doc_ids.device, dtype=torch.long
    ).view(1, -1)
    boundary = torch.ones_like(ctx_doc_ids, dtype=torch.bool)
    boundary[:, 1:] = ctx_doc_ids[:, 1:] != ctx_doc_ids[:, :-1]
    for row, row_boundary in zip(ctx_doc_ids, boundary, strict=True):
        run_ids = row[row_boundary]
        if run_ids.unique().numel() != run_ids.numel():
            raise ValueError(
                "ctx_doc_ids must contain each document in one contiguous run"
            )
    starts = torch.where(boundary, positions, torch.zeros_like(positions))
    return torch.cummax(starts, dim=1).values


def build_dspark_mla_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_length: int,
    block_size: int,
    is_causal: bool = False,
    sliding_window: int | None = None,
    context_window: int | None = None,
    ctx_doc_ids: torch.Tensor | None = None,
) -> "CudnnFlexMask":
    """Build exact interval endpoints for the DSpark context+draft mask.

    Every valid query sees one context interval and one interval inside its own
    draft block.  The common unpacked, non-windowed path uses the minimal three
    endpoints::

        [0, anchor) U [draft_begin, draft_end)

    Packed or sliding-window rows use five endpoints because their context
    interval may begin after zero::

        [0, 0) U [context_begin, anchor) U [draft_begin, draft_end)

    Invalid anchor blocks use all-zero endpoints.
    """

    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError(
            "anchor_positions and block_keep_mask must have identical [batch, anchors] shapes"
        )
    if anchor_positions.device != block_keep_mask.device:
        raise ValueError("anchor_positions and block_keep_mask must be on the same device")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if ctx_doc_ids is not None:
        expected = (anchor_positions.shape[0], context_length)
        if ctx_doc_ids.shape != expected:
            raise ValueError(f"ctx_doc_ids must have shape {expected}, got {ctx_doc_ids.shape}")
        if ctx_doc_ids.device != anchor_positions.device:
            raise ValueError("ctx_doc_ids and anchor_positions must be on the same device")

    anchors = anchor_positions.to(torch.long)
    if bool(((anchors < 0) | (anchors >= context_length)).any()):
        raise ValueError("anchor_positions must lie inside the context sequence")

    batch_size, num_anchors = anchors.shape
    query_length = num_anchors * block_size
    query_anchors = anchors.repeat_interleave(block_size, dim=1)
    valid = block_keep_mask.to(torch.bool).repeat_interleave(block_size, dim=1)
    query_offsets = (
        torch.arange(block_size, device=anchors.device, dtype=torch.long)
        .view(1, 1, block_size)
        .expand(batch_size, num_anchors, block_size)
        .reshape(batch_size, query_length)
    )
    block_ids = (
        torch.arange(num_anchors, device=anchors.device, dtype=torch.long)
        .repeat_interleave(block_size)
        .view(1, query_length)
        .expand(batch_size, -1)
    )

    context_begin = torch.zeros_like(query_anchors)
    if ctx_doc_ids is not None:
        run_starts = _document_run_starts(ctx_doc_ids)
        context_begin = torch.gather(run_starts, 1, anchors).repeat_interleave(
            block_size, dim=1
        )
        anchor_docs = torch.gather(ctx_doc_ids, 1, anchors).repeat_interleave(
            block_size, dim=1
        )
        valid = valid & (anchor_docs >= 0)
    context_end = query_anchors

    local_base = context_length + block_ids * block_size
    local_begin = local_base
    local_end = local_base + block_size

    if is_causal:
        local_end = local_base + query_offsets + 1
    if context_window is not None:
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer or null")
        context_begin = torch.maximum(context_begin, query_anchors - context_window)
    if sliding_window is not None:
        context_begin = torch.maximum(
            context_begin,
            query_anchors + query_offsets - sliding_window + 1,
        )
        context_begin = torch.minimum(context_begin, context_end)
        local_begin = local_base + torch.clamp(
            query_offsets - sliding_window + 1, min=0
        )
        local_end = torch.minimum(
            local_end,
            local_base + torch.clamp(query_offsets + sliding_window, max=block_size),
        )

    if ctx_doc_ids is None and sliding_window is None and context_window is None:
        endpoints = torch.stack((context_end, local_begin, local_end), dim=1)
    else:
        zeros = torch.zeros_like(query_anchors)
        endpoints = torch.stack(
            (zeros, context_begin, context_end, local_begin, local_end), dim=1
        )
    endpoints = torch.where(valid.unsqueeze(1), endpoints, torch.zeros_like(endpoints))
    mask_func = (
        endpoints.permute(1, 0, 2)
        .reshape(1, endpoints.shape[1], batch_size * query_length)
        .to(torch.int32)
        .contiguous()
    )
    return CudnnFlexMask(mask_func=mask_func)


@dataclass
class CudnnFlexMask:
    """Per-step endpoints plus per-geometry reusable cuDNN mask plans."""

    mask_func: torch.Tensor
    pack_gqa: bool | None = None
    fwd_variant: str | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    # PyTorch's graph/FSDP plumbing uses dataclasses.replace() while carrying
    # non-Tensor arguments. Keep the reusable plan cache constructor-visible so
    # that replacement remains valid, but exclude it from repr/equality.
    _plans: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _physical_gqa_masks: dict[tuple[Any, ...], "CudnnFlexMask"] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _joint_backward_plans: dict[tuple[Any, ...], Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        varlen = (
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
        )
        if any(value is not None for value in varlen) and not all(
            value is not None for value in varlen
        ):
            raise ValueError(
                "cuDNN THD masks require both cumulative lengths and both maxima"
            )

    def for_physical_gqa(self, ratio: int) -> "CudnnFlexMask":
        """Repeat query metadata for an equivalent Hq-to-sequence GQA layout."""

        if ratio < 1:
            raise ValueError("physical GQA ratio must be positive")
        if ratio == 1:
            return self
        if self.mask_func.shape[0] != 1:
            raise ValueError("physical GQA folding requires a head-independent mask")
        key = (
            ratio,
            self.mask_func.data_ptr(),
            self.mask_func._version,
            tuple(self.mask_func.shape),
            tuple(self.mask_func.stride()),
            self.fwd_variant,
            None if self.cu_seqlens_q is None else (self.cu_seqlens_q.data_ptr(), self.cu_seqlens_q._version),
            None if self.cu_seqlens_k is None else (self.cu_seqlens_k.data_ptr(), self.cu_seqlens_k._version),
            self.max_seqlen_q,
            self.max_seqlen_k,
        )
        physical = self._physical_gqa_masks.get(key)
        if physical is None:
            physical = CudnnFlexMask(
                mask_func=self.mask_func.repeat_interleave(ratio, dim=-1).contiguous(),
                pack_gqa=False,
                fwd_variant=self.fwd_variant,
                cu_seqlens_q=None if self.cu_seqlens_q is None else self.cu_seqlens_q * ratio,
                cu_seqlens_k=self.cu_seqlens_k,
                max_seqlen_q=None if self.max_seqlen_q is None else self.max_seqlen_q * ratio,
                max_seqlen_k=self.max_seqlen_k,
            )
            self._physical_gqa_masks[key] = physical
        return physical

    def _plan_key(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pack_gqa: bool | None,
    ) -> tuple[Any, ...]:
        return (
            self.mask_func.device,
            self.mask_func.dtype,
            tuple(self.mask_func.shape),
            tuple(self.mask_func.stride()),
            self.mask_func.data_ptr(),
            self.mask_func._version,
            q.device,
            q.dtype,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            tuple(q.stride()),
            tuple(k.stride()),
            tuple(v.stride()),
            pack_gqa,
            self.fwd_variant,
            None if self.cu_seqlens_q is None else self.cu_seqlens_q.data_ptr(),
            None if self.cu_seqlens_k is None else self.cu_seqlens_k.data_ptr(),
            None if self.cu_seqlens_q is None else self.cu_seqlens_q._version,
            None if self.cu_seqlens_k is None else self.cu_seqlens_k._version,
            self.max_seqlen_q,
            self.max_seqlen_k,
        )

    def plan_for(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pack_gqa: bool | None = None,
        build_backward: bool = True,
    ):
        effective_pack_gqa = self.pack_gqa if pack_gqa is None else pack_gqa
        key = self._plan_key(q, k, v, effective_pack_gqa) + (build_backward,)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        try:
            from cudnn.flex_attention import create_mask_plan
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "attention_backend=cudnn_flex_attention requires a cuDNN Frontend "
                "build with cudnn.flex_attention"
            ) from exc

        kwargs: dict[str, Any] = {
            "pack_gqa": effective_pack_gqa,
            "build_backward": build_backward,
        }
        if self.cu_seqlens_q is not None:
            kwargs.update(
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
            )
        if self.fwd_variant is not None:
            kwargs["_fwd_variant"] = self.fwd_variant
        plan = create_mask_plan(self.mask_func, q, k, v, **kwargs)
        self._plans[key] = plan
        return plan


def resolve_cudnn_gqa_mode(mode: str, *, physical_gqa: bool = False) -> str:
    """Resolve an explicit execution policy, preserving the legacy auto default."""
    if mode not in {"auto", "native", "external", "internal"}:
        raise ValueError(f"Unsupported cuDNN GQA mode: {mode!r}")
    return ("external" if physical_gqa else "native") if mode == "auto" else mode


def cudnn_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: CudnnFlexMask,
    scale: float,
    pack_gqa: bool | None = None,
    physical_gqa: bool = False,
    gqa_mode: str = "auto",
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute cuDNN-FE FlexAttention for native BSHD or THD tensors.

    K3's Q/K/V projections naturally produce BSHD.  Keeping that layout here
    avoids three full-tensor transpose materializations per attention layer.
    """

    if not isinstance(mask, CudnnFlexMask):
        raise TypeError("cudnn_flex_attention requires a CudnnFlexMask")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("cudnn_flex_attention requires CUDA Q/K/V tensors")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("cudnn_flex_attention requires FP16 or BF16 Q/K/V tensors")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if query.ndim not in (3, 4) or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("Q/K/V must all be rank-4 BSHD or rank-3 THD tensors")
    if query.ndim == 4:
        if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
            raise ValueError("Q/K/V batch, KV sequence, and KV head shapes must agree")
        query_heads, key_heads = query.shape[2], key.shape[2]
    else:
        if mask.cu_seqlens_q is None:
            raise ValueError("rank-3 THD Q/K/V require cumulative sequence lengths")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("THD K/V token and head shapes must agree")
        query_heads, key_heads = query.shape[1], key.shape[1]
    if query_heads % key_heads:
        raise ValueError("the number of query heads must be divisible by KV heads")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K must have the same head dimension")
    dims = (query.shape[-1], value.shape[-1])
    if dims not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            "the ASpec cuDNN path currently supports (Dqk,Dv)=(128,128) "
            f"for GQA and (192,128) for materialized MLA; got {dims}"
        )
    if dims == (192, 128) and query_heads != key_heads:
        raise ValueError("materialized MLA currently requires Hq=Hkv")

    effective_mode = resolve_cudnn_gqa_mode(gqa_mode, physical_gqa=physical_gqa)
    if effective_mode == "internal":
        from aspec.models.ops.cudnn_gqa_backward import internal_gqa_attention

        return internal_gqa_attention(
            query,
            key,
            value,
            mask=mask,
            scale=scale,
            pack_gqa=pack_gqa,
            return_lse=return_lse,
        )
    physical_gqa = effective_mode == "external"

    try:
        from cudnn.flex_attention import flex_attn_func
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "attention_backend=cudnn_flex_attention requires a cuDNN Frontend "
            "build with cudnn.flex_attention"
        ) from exc
    physical_query = query
    physical_mask = mask
    query_length = query.shape[1] if query.ndim == 4 else query.shape[0]
    if physical_gqa and query_heads != key_heads:
        physical_query = _pack_physical_gqa_query(query, key_heads)
        physical_mask = mask.for_physical_gqa(query_heads // key_heads)
        pack_gqa = False

    plan = physical_mask.plan_for(
        physical_query,
        key,
        value,
        pack_gqa=pack_gqa,
    )
    result = flex_attn_func(
        physical_query,
        key,
        value,
        mask_plan=plan,
        softmax_scale=scale,
        return_lse=return_lse,
    )
    if return_lse:
        output, lse = result
    else:
        output, lse = result, None
    if physical_query is not query:
        output = _unpack_physical_gqa_output(
            output,
            query_length=query_length,
            query_heads=query_heads,
        )
        if lse is not None:
            lse = _unpack_physical_gqa_lse(
                lse, query_length=query_length, query_heads=query_heads
            )
    return (output, lse) if return_lse else output


def _pack_physical_gqa_query(
    query: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    """Fold GQA query groups into sequence while preserving KV-head mapping."""

    is_thd = query.ndim == 3
    if is_thd:
        query = query.unsqueeze(0)
    batch, query_length, query_heads, head_dim = query.shape
    if kv_heads < 1 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    ratio = query_heads // kv_heads
    physical = (
        query.reshape(batch, query_length, kv_heads, ratio, head_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, query_length * ratio, kv_heads, head_dim)
    )
    return physical.squeeze(0) if is_thd else physical


def _unpack_physical_gqa_output(
    output: torch.Tensor,
    *,
    query_length: int,
    query_heads: int,
) -> torch.Tensor:
    """Restore physical-GQA output to its logical query-head layout."""

    is_thd = output.ndim == 3
    if is_thd:
        output = output.unsqueeze(0)
    batch, physical_length, kv_heads, value_dim = output.shape
    if query_length < 1 or physical_length % query_length:
        raise ValueError("physical GQA output length must divide by query length")
    ratio = physical_length // query_length
    if query_heads != kv_heads * ratio:
        raise ValueError("physical GQA output does not match requested query heads")
    logical = (
        output.reshape(batch, query_length, ratio, kv_heads, value_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(batch, query_length, query_heads, value_dim)
    )
    return logical.squeeze(0) if is_thd else logical


def _unpack_physical_gqa_lse(
    lse: torch.Tensor, *, query_length: int, query_heads: int
) -> torch.Tensor:
    """Restore [Hkv,T*R] / [B,Hkv,S*R], including its CP LSE gradient."""
    *batch, kv_heads, physical_length = lse.shape
    if query_length < 1 or physical_length % query_length:
        raise ValueError("physical GQA LSE length must divide by query length")
    ratio = physical_length // query_length
    if query_heads != kv_heads * ratio:
        raise ValueError("physical GQA LSE does not match requested query heads")
    return lse.reshape(*batch, kv_heads, query_length, ratio).transpose(-1, -2).reshape(
        *batch, query_heads, query_length
    )


def _prefer_thd_physical_gqa(
    query_length: int, key_length: int, query_heads: int, kv_heads: int
) -> bool:
    """Amortize query-layout work only when KV covers the expanded query axis.

    This conservative geometry policy uses the local CP partition, not the
    configured context length. Native GQA remains available for smaller KV.
    """
    return (
        kv_heads > 0
        and query_heads > kv_heads
        and query_heads % kv_heads == 0
        and key_length >= query_length * (query_heads // kv_heads)
    )


__all__ = [
    "CudnnFlexMask",
    "build_dspark_mla_mask",
    "cudnn_flex_attention",
]
