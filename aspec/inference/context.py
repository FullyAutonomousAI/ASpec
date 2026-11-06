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

"""Canonical SGLang context-length contract shared by planning and runtime."""

from __future__ import annotations

import os
from typing import Any, Mapping

SGLANG_CONTEXT_OVERRIDE_ENV = "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
SGLANG_INPUT_RESERVE_TOKENS = 6


class ContextContractError(ValueError):
    """The requested context cannot be admitted without an explicit override."""


def context_shard_bounds(
    sequence_length: int,
    rank: int,
    world_size: int,
    *,
    alignment: int = 1,
) -> tuple[int, int]:
    """Return contiguous CP ownership without splitting an aligned token group.

    Complete ``alignment``-sized groups are distributed with the existing
    floor-boundary policy.  A final incomplete group remains on the last rank,
    so no context row is duplicated or dropped.
    """

    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid context-parallel rank or world size")
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise ValueError("context shard alignment must be a positive integer")
    complete_groups = sequence_length // alignment
    start = complete_groups * rank // world_size * alignment
    end = complete_groups * (rank + 1) // world_size * alignment
    if rank == world_size - 1:
        end = sequence_length
    return start, end


def context_shard_lengths(
    sequence_length: int,
    world_size: int,
    *,
    alignment: int = 1,
) -> list[int]:
    """Return every rank's exact aligned context ownership length."""

    return [
        end - start
        for rank in range(world_size)
        for start, end in [
            context_shard_bounds(
                sequence_length, rank, world_size, alignment=alignment
            )
        ]
    ]


def sglang_training_context_length(max_sequence_length: int | None) -> int | None:
    """Return the SGLang context ceiling required by one training input."""
    if max_sequence_length is None:
        return None
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    return max_sequence_length + SGLANG_INPUT_RESERVE_TOKENS


def _positive_integer(value: Any, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContextContractError(
            f"{source} must be a positive integer, got {value!r}"
        )
    return value


def _model_context_limits(
    model_config: Mapping[str, Any] | None,
    native_context_length: int | None,
) -> dict[str, Any]:
    text_config = None
    if model_config is not None:
        nested = model_config.get("text_config")
        text_config = nested if isinstance(nested, Mapping) else model_config

    config_limit = None
    config_limit_source = None
    if text_config is not None and text_config.get("max_position_embeddings") is not None:
        config_limit_source = (
            "model.text_config.max_position_embeddings"
            if text_config is not model_config
            else "model.max_position_embeddings"
        )
        config_limit = _positive_integer(
            text_config["max_position_embeddings"], config_limit_source
        )

    native_source = None
    if native_context_length is not None:
        native_limit = _positive_integer(
            native_context_length, "model.native_context_length"
        )
        native_source = "model.native_context_length"
    else:
        native_limit = config_limit
        native_source = config_limit_source

    extension = None
    if text_config is not None:
        rope = text_config.get("rope_scaling") or text_config.get("rope_parameters")
        if isinstance(rope, Mapping):
            rope_type = rope.get("rope_type", rope.get("type"))
            if rope_type == "yarn":
                original = _positive_integer(
                    rope.get("original_max_position_embeddings"),
                    "model rope_scaling.original_max_position_embeddings",
                )
                factor = rope.get("factor")
                if isinstance(factor, bool) or not isinstance(factor, (int, float)):
                    raise ContextContractError(
                        "model rope_scaling.factor must be a positive number"
                    )
                if factor <= 0 or not float(factor).is_integer():
                    raise ContextContractError(
                        "model rope_scaling.factor must be a positive integer factor"
                    )
                extension = {
                    "type": "yarn",
                    "factor": float(factor),
                    "original_context_length": original,
                    "declared_context_limit": original * int(factor),
                    "source": "model rope_scaling",
                }
                if native_context_length is None:
                    native_limit = original
                    native_source = (
                        "model rope_scaling.original_max_position_embeddings"
                    )

    admitted_training_limit = (
        extension["declared_context_limit"] if extension else native_limit
    )
    # A declared RoPE extension admits the training workload, but it does not
    # necessarily change the limit enforced by the serving backend.  In
    # particular, SGLang keeps using max_position_embeddings when YaRN carries
    # original_max_position_embeddings and therefore still requires its
    # explicit context-length override above that value.
    backend_declared_limit = config_limit or native_limit
    return {
        "model_native_context_limit": native_limit,
        "model_native_context_limit_source": native_source,
        "model_config_context_limit": config_limit,
        "model_config_context_limit_source": config_limit_source,
        "model_context_extension": extension,
        "admitted_training_context_limit": admitted_training_limit,
        "backend_declared_context_limit": backend_declared_limit,
    }


def compile_sglang_context_contract(
    *,
    max_sequence_length: int,
    context_parallel_size: int,
    model_config: Mapping[str, Any] | None,
    native_context_length: int | None,
    allow_context_length_override: bool,
    context_shard_alignment: int = 1,
) -> dict[str, Any]:
    """Resolve static context quantities and reject an undeclared override."""
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise ContextContractError(
            "training.max_seq_length must be a positive integer"
        )
    if max_sequence_length < 1:
        raise ContextContractError("training.max_seq_length must be positive")
    if context_parallel_size < 1:
        raise ContextContractError(
            "training.draft_context_parallel_size must be positive"
        )

    backend_context_length = sglang_training_context_length(max_sequence_length)
    model_limits = _model_context_limits(model_config, native_context_length)
    admitted_training_limit = model_limits["admitted_training_context_limit"]
    if (
        admitted_training_limit is not None
        and max_sequence_length > admitted_training_limit
    ):
        raise ContextContractError(
            f"training.max_seq_length={max_sequence_length} exceeds the model's "
            f"admitted training context limit {admitted_training_limit}; a backend "
            "context override cannot replace a declared RoPE/context extension"
        )

    backend_declared_limit = model_limits["backend_declared_context_limit"]
    override_required = (
        None
        if backend_declared_limit is None
        else backend_context_length > backend_declared_limit
    )
    if override_required and not allow_context_length_override:
        raise ContextContractError(
            "SGLang requires backend context_length="
            f"{backend_context_length} for training.max_seq_length="
            f"{max_sequence_length} plus {SGLANG_INPUT_RESERVE_TOKENS} reserve tokens, "
            f"which exceeds the backend-declared limit {backend_declared_limit}; set "
            "inference.sglang.allow_context_length_override=true only after validating "
            "this model/backend combination"
        )

    shard_lengths = context_shard_lengths(
        max_sequence_length,
        context_parallel_size,
        alignment=context_shard_alignment,
    )
    override_effective = bool(
        allow_context_length_override and override_required is not False
    )
    return {
        "requested_max_sequence_length": max_sequence_length,
        "effective_training_max_sequence_length": max_sequence_length,
        "target_input_reserve_tokens": SGLANG_INPUT_RESERVE_TOKENS,
        "backend_context_length": backend_context_length,
        **model_limits,
        "model_limit_state": (
            "resolved"
            if admitted_training_limit is not None
            else "pending_backend_probe"
        ),
        "context_override": {
            "allowed": bool(allow_context_length_override),
            "required": override_required,
            "effective": override_effective,
            "mechanism": SGLANG_CONTEXT_OVERRIDE_ENV,
        },
        "training_sharding": {
            "context_parallel_size": context_parallel_size,
            "context_shard_alignment": context_shard_alignment,
            "shard_lengths_at_max": shard_lengths,
            "sequence_padding_tokens_at_max": 0,
        },
    }


def configure_sglang_context_override(contract: Mapping[str, Any]) -> bool:
    """Make the canonical Plan, rather than the launcher environment, authoritative."""
    override = contract.get("context_override")
    enabled = bool(isinstance(override, Mapping) and override.get("effective"))
    if enabled:
        os.environ[SGLANG_CONTEXT_OVERRIDE_ENV] = "1"
    else:
        os.environ.pop(SGLANG_CONTEXT_OVERRIDE_ENV, None)
    return enabled


def compile_offline_context_contract(
    *, max_sequence_length, context_parallel_size, model_config,
    native_context_length, allow_context_length_override, context_shard_alignment=1,
):
    """Plan stored sequence bounds without a live backend token pool."""
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ContextContractError("training.max_seq_length must be a positive integer")
    if context_parallel_size != 1:
        raise ContextContractError("Offline replay currently requires CP1")
    limits = _model_context_limits(model_config, native_context_length)
    admitted = limits["admitted_training_context_limit"]
    if admitted is not None and max_sequence_length > admitted:
        raise ContextContractError("Offline training.max_seq_length exceeds the model context limit")
    return {
        "requested_max_sequence_length": max_sequence_length,
        "effective_training_max_sequence_length": max_sequence_length,
        "target_input_reserve_tokens": 0,
        "backend_context_length": None,
        **limits,
        "model_limit_state": "resolved" if admitted is not None else "recorded_sequences_only",
        "context_override": {"allowed": False, "required": False, "effective": False, "mechanism": None},
        "training_sharding": {
            "context_parallel_size": 1,
            "context_shard_alignment": context_shard_alignment,
            "shard_lengths_at_max": context_shard_lengths(max_sequence_length, 1, alignment=context_shard_alignment),
            "sequence_padding_tokens_at_max": 0,
        },
    }
