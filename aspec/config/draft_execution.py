# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Canonical Draft attention and context-parallel execution capabilities."""

from __future__ import annotations

from dataclasses import dataclass

FA4_ATTENTION_BACKEND = "fa4"
LEGACY_DIRECT_CP_FA4_BACKEND = "direct_cp_fa4"
CUDNN_FLEX_ATTENTION_BACKEND = "cudnn_flex_attention"


@dataclass(frozen=True)
class DraftExecutionCapability:
    """Resolved execution dimensions consumed by runtime components."""

    attention_backend: str
    context_parallel_size: int
    fa4_enabled: bool
    direct_cp_backend: bool
    context_parallel_enabled: bool
    feature_layout: str
    legacy_backend_alias: bool


def normalize_attention_backend(attention_backend: str | None) -> str | None:
    if attention_backend == LEGACY_DIRECT_CP_FA4_BACKEND:
        return FA4_ATTENTION_BACKEND
    return attention_backend


def is_fa4_backend(attention_backend: str | None) -> bool:
    return normalize_attention_backend(attention_backend) == FA4_ATTENTION_BACKEND


def is_cudnn_flex_backend(attention_backend: str | None) -> bool:
    return normalize_attention_backend(attention_backend) == CUDNN_FLEX_ATTENTION_BACKEND


def is_direct_cp_backend(attention_backend: str | None) -> bool:
    """Return whether the backend implements ASpec's direct CP contract."""

    backend = normalize_attention_backend(attention_backend)
    return backend in {FA4_ATTENTION_BACKEND, CUDNN_FLEX_ATTENTION_BACKEND}


def resolve_draft_execution_capability(
    attention_backend: str | None,
    context_parallel_size: int = 1,
) -> DraftExecutionCapability:
    """Resolve independent backend and CP dimensions and reject illegal pairs."""

    legacy_alias = attention_backend == LEGACY_DIRECT_CP_FA4_BACKEND
    backend = normalize_attention_backend(attention_backend)
    try:
        cp_size = int(context_parallel_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "draft_context_parallel_size must be a positive integer, got "
            f"{context_parallel_size!r}"
        ) from exc
    if cp_size <= 0 or isinstance(context_parallel_size, bool) or (
        isinstance(context_parallel_size, float) and context_parallel_size != cp_size
    ):
        raise ValueError(
            f"draft_context_parallel_size must be a positive integer, got {cp_size}"
        )

    fa4_enabled = backend == FA4_ATTENTION_BACKEND
    direct_cp_backend = is_direct_cp_backend(backend)
    if cp_size > 1 and not direct_cp_backend:
        raise NotImplementedError(
            "draft context parallelism requires attention_backend=fa4 or "
            "attention_backend=cudnn_flex_attention"
        )

    return DraftExecutionCapability(
        attention_backend=backend,
        context_parallel_size=cp_size,
        fa4_enabled=fa4_enabled,
        direct_cp_backend=direct_cp_backend,
        context_parallel_enabled=cp_size > 1,
        feature_layout="context_sharded" if cp_size > 1 else "full",
        legacy_backend_alias=legacy_alias,
    )
