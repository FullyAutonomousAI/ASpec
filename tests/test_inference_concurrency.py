# Copyright (C) 2026 Tencent. All rights reserved.
# Adapted from the retained AngelSpec source snapshot 0966eb139d1c4054767e4a37bfc8a33db5989a72
# tests/test_inference_manager_failures.py. Original portions remain Apache-2.0;
# see licenses/AngelSpec-LICENSE. ASpec extends contract and lifecycle coverage.
#
# ASpec modifications:
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

"""Unit coverage for bounded, fair Target inference concurrency."""

import asyncio
import os
import time

import pytest

from aspec.controller.inference_manager import (
    EnginePool,
    effective_prompt_buffer_watermark,
    producer_batch_rate_metrics,
)
from aspec.inference.concurrency import (
    effective_engine_concurrency,
    required_target_prefill_tokens,
)
from aspec.inference.context import (
    SGLANG_CONTEXT_OVERRIDE_ENV,
    compile_sglang_context_contract,
    configure_sglang_context_override,
    sglang_training_context_length,
)
from aspec.inference.engine.sgl_engine import (
    sglang_training_max_prefill_tokens,
    validate_sglang_profiled_capacity,
)
from aspec.locality import (
    allocate_node_ready_watermarks,
    allocate_node_sample_budgets,
)


def test_e2e_producer_metric_is_not_reported_as_independent_capacity():
    metrics = producer_batch_rate_metrics(
        [(4, 2.0, time.time())],
        perf_window_seconds=10.0,
        max_concurrent=3,
    )

    assert metrics["perf/producer_batch_rate_estimate"] == 6.0
    assert "perf/infer_capacity" not in metrics


def test_prompt_buffer_watermark_can_feed_engine_concurrency():
    assert (
        effective_prompt_buffer_watermark(
            configured=2,
            batch_size=2,
            max_concurrent=5,
        )
        == 10
    )
    assert (
        effective_prompt_buffer_watermark(
            configured=32,
            batch_size=2,
            max_concurrent=10,
        )
        == 32
    )


def test_node_ready_watermarks_preserve_total_and_follow_draft_demand():
    assert allocate_node_ready_watermarks(
        10, {"node-a": 1, "node-b": 3}
    ) == {"node-a": 3, "node-b": 7}

    with pytest.raises(ValueError, match="one credit per node"):
        allocate_node_ready_watermarks(1, {"node-a": 1, "node-b": 1})


def test_finite_node_sample_budget_preserves_complete_local_requests():
    assert allocate_node_sample_budgets(
        16,
        {"node-a": 1, "node-b": 3},
        request_batch_size=2,
    ) == {"node-a": 4, "node-b": 12}

    with pytest.raises(ValueError, match="complete Target requests"):
        allocate_node_sample_budgets(
            10,
            {"node-a": 1, "node-b": 1},
            request_batch_size=2,
        )


def test_engine_concurrency_respects_source_ring_capacity():
    assert effective_engine_concurrency(
        configured=2,
        batch_size=2,
        source_slots=2,
    ) == 1
    assert effective_engine_concurrency(
        configured=3,
        batch_size=1,
        source_slots=6,
        slots_per_sample=2,
    ) == 3
    assert effective_engine_concurrency(
        configured=3,
        batch_size=2,
        source_slots=None,
    ) == 3
    with pytest.raises(ValueError, match="cannot hold one inference request"):
        effective_engine_concurrency(
            configured=2,
            batch_size=2,
            source_slots=3,
            slots_per_sample=2,
        )


def test_target_prefill_token_capacity_uses_effective_concurrency():
    assert required_target_prefill_tokens(
        max_sequence_length=262144,
        batch_size=2,
        max_concurrent_batches=1,
    ) == 524544


def test_sglang_training_capacity_guards_long_prefill():
    assert sglang_training_context_length(262144) == 262150
    assert sglang_training_max_prefill_tokens(None) == (1 << 31) - 1
    assert sglang_training_max_prefill_tokens(-1) == (1 << 31) - 1
    with pytest.raises(RuntimeError, match="profiled max_total_tokens"):
        validate_sglang_profiled_capacity(262143, 262400)


def test_sglang_context_override_is_controlled_by_the_plan(monkeypatch):
    contract = compile_sglang_context_contract(
        max_sequence_length=262144,
        context_parallel_size=2,
        model_config={
            "text_config": {"max_position_embeddings": 262144}
        },
        native_context_length=262144,
        allow_context_length_override=True,
    )

    monkeypatch.setenv(SGLANG_CONTEXT_OVERRIDE_ENV, "unexpected-launcher-value")
    assert configure_sglang_context_override(contract) is True
    assert os.environ[SGLANG_CONTEXT_OVERRIDE_ENV] == "1"

    in_range = compile_sglang_context_contract(
        max_sequence_length=131072,
        context_parallel_size=1,
        model_config={
            "text_config": {"max_position_embeddings": 262144}
        },
        native_context_length=262144,
        allow_context_length_override=False,
    )
    assert configure_sglang_context_override(in_range) is False
    assert SGLANG_CONTEXT_OVERRIDE_ENV not in os.environ


def test_yarn_extension_is_distinct_from_backend_override():
    contract = compile_sglang_context_contract(
        max_sequence_length=131072,
        context_parallel_size=1,
        model_config={
            "max_position_embeddings": 40960,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        },
        native_context_length=32768,
        allow_context_length_override=True,
    )

    assert contract["model_native_context_limit"] == 32768
    assert contract["model_context_extension"]["declared_context_limit"] == 131072
    assert contract["admitted_training_context_limit"] == 131072
    assert contract["backend_declared_context_limit"] == 40960
    assert contract["context_override"]["required"] is True
    assert contract["context_override"]["effective"] is True


def test_engine_pool_leases_distinct_slots_and_recovers_after_failure():
    async def scenario():
        pool = EnginePool(["engine-0", "engine-1"], max_concurrent_per_engine=1)
        assert pool.available == 2
        with pytest.raises(RuntimeError, match="request failed"):
            async with pool.lease() as first:
                async with pool.lease() as second:
                    assert {first, second} == {"engine-0", "engine-1"}
                    assert pool.available == 0
                    raise RuntimeError("request failed")
        assert pool.available == 2

    asyncio.run(scenario())


def test_node_local_engine_pool_uses_observed_placement_not_handle_order():
    engines = ["node-1-a", "node-0-a", "node-1-b", "node-0-b"]
    engine_nodes = ["node-1", "node-0", "node-1", "node-0"]

    async def scenario():
        pool = EnginePool(
            engines,
            engine_nodes=engine_nodes,
            planned_nodes=["node-0", "node-1"],
        )
        leased = []
        async with pool.lease() as first:
            leased.append(first)
            async with pool.lease() as second:
                leased.append(second)
                async with pool.lease() as third:
                    leased.append(third)
                    async with pool.lease() as fourth:
                        leased.append(fourth)
        assert leased == [
            "node-0-a",
            "node-1-a",
            "node-0-b",
            "node-1-b",
        ]

    asyncio.run(scenario())


def test_node_local_engine_pool_rotation_ignores_completion_order():
    async def scenario():
        engines = ["node-0-a", "node-0-b", "node-1-a", "node-1-b"]
        pool = EnginePool(
            engines,
            engine_nodes=["node-0", "node-0", "node-1", "node-1"],
            planned_nodes=["node-0", "node-1"],
        )

        first_round = [pool.lease() for _ in range(4)]
        leased = [await context.__aenter__() for context in first_round]
        assert leased == ["node-0-a", "node-1-a", "node-0-b", "node-1-b"]

        # Simulate both node-0 requests completing before either node-1
        # request.  A global return FIFO would schedule node 0 twice next.
        for index in (0, 2, 1, 3):
            await first_round[index].__aexit__(None, None, None)

        second_round = [pool.lease() for _ in range(4)]
        leased = [await context.__aenter__() for context in second_round]
        assert leased == ["node-0-a", "node-1-a", "node-0-b", "node-1-b"]
        for context in second_round:
            await context.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_node_local_engine_pool_rejects_placement_outside_plan():
    with pytest.raises(ValueError, match="do not match the Plan"):
        EnginePool(
            ["engine-0", "engine-1", "engine-2"],
            engine_nodes=["node-0", "node-0", "node-2"],
            planned_nodes=["node-0", "node-1"],
        )
