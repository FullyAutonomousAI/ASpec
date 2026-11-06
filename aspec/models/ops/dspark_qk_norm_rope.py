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

"""Autograd adapter for ASpec's DSpark-specific CuTeDSL Q/K kernels."""

from __future__ import annotations

import os
from typing import Any

import torch

_IMPORT_ERROR: Exception | None = None
try:
    from aspec.kernels.cutedsl.dspark_qk_norm_rope import (
        compile_dspark_qk_norm_rope,
        run_dspark_qk_norm_rope,
    )
    from aspec.kernels.cutedsl.dspark_qk_norm_rope_bwd import (
        compile_dspark_qk_norm_rope_bwd,
        dspark_n_ctas_for,
        run_dspark_qk_norm_rope_bwd,
    )
except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc


_FWD_RECIPES: dict[tuple[Any, ...], Any] = {}
_BWD_RECIPES: dict[tuple[Any, ...], Any] = {}


def dspark_qk_norm_rope_available() -> bool:
    """Whether the ASpec CuTeDSL runtime can compile the DSpark specialization."""

    return (
        _IMPORT_ERROR is None
        and os.getenv("ASPEC_DISABLE_DSPARK_QK_KERNEL") != "1"
        and os.getenv("ASPEC_DISABLE_CUDNN_DSPARK_QK_KERNEL") != "1"
    )


def _recipe_key(
    device: torch.device,
    dtype: torch.dtype,
    h_q: int,
    h_kv: int,
    d: int,
    eps: float,
    batch_one: bool,
) -> tuple[Any, ...]:
    return (
        device.index,
        dtype,
        int(h_q),
        int(h_kv),
        int(d),
        float(eps),
        bool(batch_one),
    )


def _forward_recipe(
    device: torch.device,
    dtype: torch.dtype,
    h_q: int,
    h_kv: int,
    d: int,
    eps: float,
    batch_one: bool,
):
    key = _recipe_key(device, dtype, h_q, h_kv, d, eps, batch_one)
    recipe = _FWD_RECIPES.get(key)
    if recipe is None:
        with torch.cuda.device(device):
            recipe = compile_dspark_qk_norm_rope(
                dtype=dtype,
                h_q=h_q,
                h_kv=h_kv,
                d=d,
                rope_dim=d,
                eps=eps,
                defer_secondary_loads=False,
                batch_one=batch_one,
            )
        _FWD_RECIPES[key] = recipe
    return recipe


def _backward_recipe(
    device: torch.device,
    dtype: torch.dtype,
    h_q: int,
    h_kv: int,
    d: int,
    eps: float,
    batch_one: bool,
):
    key = _recipe_key(device, dtype, h_q, h_kv, d, eps, batch_one)
    recipe = _BWD_RECIPES.get(key)
    if recipe is None:
        with torch.cuda.device(device):
            recipe = compile_dspark_qk_norm_rope_bwd(
                dtype=dtype,
                h_q=h_q,
                h_kv=h_kv,
                d=d,
                rope_dim=d,
                eps=eps,
                batch_one=batch_one,
            )
        _BWD_RECIPES[key] = recipe
    return recipe


def _require_compact(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous for the DSpark fused Q/K kernel, "
            f"got stride {tensor.stride()}"
        )


class _DSparkQkNormRope(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        context_key: torch.Tensor,
        draft_key: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        draft_position_ids: torch.Tensor,
        variance_epsilon: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "the ASpec CuTeDSL dependencies cannot provide the "
                "DSpark Q/K norm+RoPE specialization"
            ) from _IMPORT_ERROR
        for name, tensor in (
            ("query", query),
            ("context_key", context_key),
            ("draft_key", draft_key),
            ("query_position_ids", query_position_ids),
            ("context_position_ids", context_position_ids),
            ("draft_position_ids", draft_position_ids),
        ):
            _require_compact(name, tensor)
        if query.device.type != "cuda":
            raise ValueError("the DSpark fused Q/K kernel requires CUDA tensors")
        if query.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"the DSpark fused Q/K kernel serves bf16/f16, got {query.dtype}")
        if context_key.dtype != query.dtype or draft_key.dtype != query.dtype:
            raise ValueError("query, context_key, and draft_key must share one dtype")
        if query_norm_weight.dtype != query.dtype or key_norm_weight.dtype != query.dtype:
            raise ValueError("Q/K norm weights must share the activation dtype")

        batch, draft_len, query_width = query.shape
        context_len = int(context_key.shape[1])
        if tuple(draft_key.shape[:2]) != (batch, draft_len):
            raise ValueError("draft_key must share query batch and draft length")
        if int(context_key.shape[0]) != batch:
            raise ValueError("context_key must share the query batch")
        if query_width != num_heads * head_dim:
            raise ValueError("query width does not match num_heads * head_dim")
        if (
            int(context_key.shape[2]) != num_kv_heads * head_dim
            or int(draft_key.shape[2]) != num_kv_heads * head_dim
        ):
            raise ValueError("K width does not match num_kv_heads * head_dim")

        query_3d = query.view(batch * draft_len, num_heads, head_dim)
        context_key_3d = context_key.view(batch * context_len, num_kv_heads, head_dim)
        draft_key_3d = draft_key.view(batch * draft_len, num_kv_heads, head_dim)
        position_q = query_position_ids.view(-1)
        position_context = context_position_ids.view(-1)
        position_draft = draft_position_ids.view(-1)
        cos_2d = cos.squeeze(0).squeeze(0)
        sin_2d = sin.squeeze(0).squeeze(0)
        _require_compact("cos", cos_2d)
        _require_compact("sin", sin_2d)

        query_out = torch.empty_like(query_3d)
        key_out = torch.empty(
            batch * (context_len + draft_len),
            num_kv_heads,
            head_dim,
            device=query.device,
            dtype=query.dtype,
        )
        rstd_q = torch.empty(
            batch * draft_len,
            num_heads,
            device=query.device,
            dtype=torch.float32,
        )
        rstd_k = torch.empty(
            batch * (context_len + draft_len),
            num_kv_heads,
            device=query.device,
            dtype=torch.float32,
        )
        recipe = _forward_recipe(
            query.device,
            query.dtype,
            num_heads,
            num_kv_heads,
            head_dim,
            variance_epsilon,
            batch == 1,
        )
        stream = torch.cuda.current_stream(query.device).cuda_stream
        run_dspark_qk_norm_rope(
            recipe,
            query_3d,
            context_key_3d,
            draft_key_3d,
            query_out,
            key_out,
            query_norm_weight,
            key_norm_weight,
            cos_2d,
            sin_2d,
            position_q,
            position_context,
            position_draft,
            rstd_q,
            rstd_k,
            batch=batch,
            context_len=context_len,
            draft_len=draft_len,
            stream=stream,
        )
        ctx.save_for_backward(
            query_3d,
            context_key_3d,
            draft_key_3d,
            query_norm_weight,
            key_norm_weight,
            cos_2d,
            sin_2d,
            position_q,
            position_context,
            position_draft,
            rstd_q,
            rstd_k,
        )
        ctx.batch = batch
        ctx.context_len = context_len
        ctx.draft_len = draft_len
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.variance_epsilon = float(variance_epsilon)
        return (
            query_out.view(batch, draft_len, num_heads, head_dim),
            key_out.view(batch, context_len + draft_len, num_kv_heads, head_dim),
        )

    @staticmethod
    def backward(ctx, grad_query: torch.Tensor, grad_key: torch.Tensor):
        (
            query,
            context_key,
            draft_key,
            query_norm_weight,
            key_norm_weight,
            cos,
            sin,
            position_q,
            position_context,
            position_draft,
            rstd_q,
            rstd_k,
        ) = ctx.saved_tensors
        grad_query = grad_query.contiguous().view(
            ctx.batch * ctx.draft_len,
            ctx.num_heads,
            ctx.head_dim,
        )
        grad_key = grad_key.contiguous().view(
            ctx.batch * (ctx.context_len + ctx.draft_len),
            ctx.num_kv_heads,
            ctx.head_dim,
        )
        out_query = torch.empty_like(query)
        out_context_key = torch.empty_like(context_key)
        out_draft_key = torch.empty_like(draft_key)
        recipe = _backward_recipe(
            query.device,
            query.dtype,
            ctx.num_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
            ctx.variance_epsilon,
            ctx.batch == 1,
        )
        n_q, n_k = dspark_n_ctas_for(
            recipe,
            ctx.batch * ctx.draft_len,
            ctx.batch * (ctx.context_len + ctx.draft_len),
        )
        dw_partials_q = torch.empty(
            n_q,
            ctx.head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        dw_partials_k = torch.empty(
            n_k,
            ctx.head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        dw_q = torch.empty(
            ctx.head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        dw_k = torch.empty_like(dw_q)
        stream = torch.cuda.current_stream(query.device).cuda_stream
        run_dspark_qk_norm_rope_bwd(
            recipe,
            grad_query,
            grad_key,
            query,
            context_key,
            draft_key,
            rstd_q,
            rstd_k,
            query_norm_weight,
            key_norm_weight,
            cos,
            sin,
            position_q,
            position_context,
            position_draft,
            out_query,
            out_context_key,
            out_draft_key,
            dw_partials_q,
            dw_partials_k,
            dw_q,
            dw_k,
            batch=ctx.batch,
            context_len=ctx.context_len,
            draft_len=ctx.draft_len,
            stream=stream,
        )
        return (
            out_query.view(ctx.batch, ctx.draft_len, -1),
            out_context_key.view(ctx.batch, ctx.context_len, -1),
            out_draft_key.view(ctx.batch, ctx.draft_len, -1),
            dw_q,
            dw_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dspark_qk_norm_rope(
    query: torch.Tensor,
    context_key: torch.Tensor,
    draft_key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    variance_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position_ids: torch.Tensor,
    context_position_ids: torch.Tensor,
    draft_position_ids: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Q/K RMSNorm, split-K join and indexed RoPE for DSpark training."""

    return _DSparkQkNormRope.apply(
        query,
        context_key,
        draft_key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        query_position_ids,
        context_position_ids,
        draft_position_ids,
        variance_epsilon,
        num_heads,
        num_kv_heads,
        head_dim,
    )
