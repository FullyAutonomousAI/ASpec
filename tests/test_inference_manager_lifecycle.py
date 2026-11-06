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

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from aspec.controller.inference_manager import (
    AsyncInferenceManager,
    EnginePool,
    MetricsCollector,
    remaining_sample_budget,
    resolve_training_sample_budget,
    should_dispatch_partial_batch,
)
from aspec.utils.types import InferenceInput
from aspec.work_lifecycle import EXECUTION_ID_KEY, build_work_identity


class _AsyncRemote:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _manager():
    manager_class = AsyncInferenceManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager._metrics = MetricsCollector()
    manager._running = True
    manager._task_nodes = {}
    manager._node_ready_watermarks = {}
    manager._node_dispatch_demand = {}
    manager._sample_budget_by_node = {}
    manager._scheduled_samples_by_node = {}
    manager._batch_size = 1
    manager._phase_after_eval = None
    manager.controller = SimpleNamespace(
        push_inference_results=_AsyncRemote(),
        set_inference_error=_AsyncRemote(),
    )
    return manager


def test_forward_results_preserves_success_prefix_then_fails_closed():
    async def scenario():
        manager = _manager()
        first = InferenceInput(
            data_id="first",
            prompt="first",
            metadata=build_work_identity(split="train", epoch=0, source_id="first"),
        )
        second = InferenceInput(
            data_id="second",
            prompt="second",
            metadata=build_work_identity(split="train", epoch=0, source_id="second"),
        )

        with pytest.raises(RuntimeError, match="lost required work"):
            await manager._forward_results(
                [
                    (first, {"mooncake_key": "key-first"}),
                    (second, ValueError("backend rejected request")),
                ]
            )

        forwarded = manager.controller.push_inference_results.calls[0][0][0]
        assert [item.data_id for item in forwarded] == ["first"]
        assert manager.controller.set_inference_error.calls
        assert manager._running is False

    asyncio.run(scenario())


def test_engine_output_cannot_override_scheduler_owned_identity():
    manager = _manager()
    metadata = build_work_identity(split="train", epoch=0, source_id="row")
    entry = InferenceInput(data_id="row", prompt="row", metadata=metadata)

    with pytest.raises(ValueError, match="scheduler-owned"):
        manager._parse_engine_output(
            entry,
            {
                "mooncake_key": "key-row",
                "metadata": {EXECUTION_ID_KEY: "backend-execution"},
            },
        )


def test_engine_output_may_repeat_same_identity_and_add_backend_facts():
    manager = _manager()
    metadata = build_work_identity(split="train", epoch=0, source_id="row")
    entry = InferenceInput(data_id="row", prompt="row", metadata=metadata)

    result = manager._parse_engine_output(
        entry,
        {
            "mooncake_key": "key-row",
            "metadata": {**metadata, "backend_rank": 3},
        },
    )

    assert result.metadata == {**metadata, "backend_rank": 3}


def test_invalid_engine_output_is_not_silently_omitted():
    async def scenario():
        manager = _manager()
        entry = InferenceInput(data_id="row", prompt="row", metadata={})

        with pytest.raises(RuntimeError, match="invalid engine output"):
            await manager._forward_results([(entry, {})])

        assert not manager.controller.push_inference_results.calls

    asyncio.run(scenario())


def test_run_propagates_prompt_refill_failure():
    async def scenario():
        manager = _manager()
        manager._pending_tasks = set()

        class _Engines(list):
            max_concurrent = 1

        manager._engines = _Engines([object()])

        async def fail_refill():
            raise RuntimeError("prompt refill failed")

        async def no_dispatch():
            return None

        async def no_drain():
            return None

        manager._continuous_refill = fail_refill
        manager._dispatch_ready_batches = no_dispatch
        manager._drain = no_drain

        with pytest.raises(RuntimeError, match="prompt refill failed"):
            await manager.run()

        assert manager._running is False

    asyncio.run(scenario())


def test_fixed_workload_has_exact_sample_budget_but_packing_does_not():
    assert resolve_training_sample_budget(3, 8, packing_enabled=False) == 24
    assert resolve_training_sample_budget(3, 8, packing_enabled=True) is None
    assert resolve_training_sample_budget(
        3, 8, packing_enabled=True, source_work_budget=17
    ) == 17
    with pytest.raises(ValueError, match="positive"):
        resolve_training_sample_budget(
            3, 8, packing_enabled=True, source_work_budget=0
        )
    assert remaining_sample_budget(24, 16) == 8
    assert remaining_sample_budget(24, 24) == 0


def test_producer_profile_budget_must_drain_before_training_budget_starts():
    manager = _manager()
    manager._budget_phase = "producer_profile"
    manager._sample_budget = 6
    manager._training_sample_budget = 24
    manager._scheduled_samples = 5
    manager._pending_tasks = set()
    manager._inflight_samples = 0
    manager._prompt_buffer = deque()
    manager._source_dry = True
    manager._batch_times = deque([(1, 1.0, 1.0)])
    manager._peak_reserved_samples = 3
    manager._capacity_rejections = 2

    with pytest.raises(RuntimeError, match="before producer work drains"):
        manager.start_training_budget()

    manager._scheduled_samples = 6
    transition = manager.start_training_budget()

    assert transition == {"phase": "training", "sample_budget": 24}
    assert manager._sample_budget == 24
    assert manager._scheduled_samples == 0
    assert not manager._batch_times
    assert manager._source_dry is False


def test_eval_budget_is_isolated_and_restores_full_training_budget():
    manager = _manager()
    manager._budget_phase = "training"
    manager._sample_budget = 24
    manager._training_sample_budget = 24
    manager._scheduled_samples = 0
    manager._pending_tasks = set()
    manager._inflight_samples = 0
    manager._prompt_buffer = deque()
    manager._source_dry = False
    manager._batch_times = deque()
    manager._peak_reserved_samples = 0
    manager._peak_inflight_samples = 0
    manager._peak_admitted_ready_samples = 0
    manager._capacity_rejections = 0
    manager._node_dispatch_demand = {}
    manager._batch_size = 1
    manager._phase_after_eval = None

    transition = manager.start_eval_budget(40)
    assert transition == {"phase": "eval", "sample_budget": 40}

    manager._scheduled_samples = 40
    restored = manager.finish_eval_budget()

    assert restored == {"phase": "training", "sample_budget": 24}
    assert manager._scheduled_samples == 0
    assert manager._phase_after_eval is None


def test_eval_budget_cannot_restore_until_all_eval_work_drains():
    manager = _manager()
    manager._budget_phase = "training"
    manager._sample_budget = 8
    manager._training_sample_budget = 8
    manager._scheduled_samples = 0
    manager._pending_tasks = set()
    manager._inflight_samples = 0
    manager._prompt_buffer = deque()
    manager._source_dry = False
    manager._batch_times = deque()
    manager._peak_reserved_samples = 0
    manager._peak_inflight_samples = 0
    manager._peak_admitted_ready_samples = 0
    manager._capacity_rejections = 0
    manager._node_dispatch_demand = {}
    manager._batch_size = 1
    manager._phase_after_eval = None

    manager.start_eval_budget(16)
    manager._scheduled_samples = 15

    with pytest.raises(RuntimeError, match="cannot finish eval before its work drains"):
        manager.finish_eval_budget()


def test_final_budget_tail_dispatches_without_waiting_for_source_dry():
    assert should_dispatch_partial_batch(
        buffered=2,
        batch_size=4,
        pending=0,
        max_concurrent=1,
        remaining_budget=2,
        source_dry=False,
    )

    async def scenario():
        manager = _manager()
        manager._prompt_buffer = deque(
            InferenceInput(data_id=str(index), prompt=str(index), metadata={})
            for index in range(2)
        )
        manager._batch_size = 4
        manager._pending_tasks = set()
        manager._task_sizes = {}
        manager._task_nodes = {}
        manager._engines = EnginePool(["engine"])
        manager._node_ready_watermarks = {}
        manager._sample_budget = 6
        manager._scheduled_samples = 4
        manager._source_dry = False

        async def reserve(size, *, node=None):
            assert node is None
            return size == 2

        async def dispatch(entries, *, engine_lease=None):
            engine_lease.release()
            return entries

        manager._reserve_pool_capacity = reserve
        manager._dispatch_batch = dispatch

        await manager._dispatch_ready_batches()

        assert manager._scheduled_samples == 6
        assert not manager._prompt_buffer
        assert len(manager._pending_tasks) == 1
        task = next(iter(manager._pending_tasks))
        assert manager._task_sizes[task] == 2
        await task

    asyncio.run(scenario())


def test_admission_uses_ready_queue_not_lifecycle_residency():
    async def scenario():
        manager = _manager()
        ready_pool = _AsyncRemote(4)
        resident_pool = _AsyncRemote(12)
        manager.controller = SimpleNamespace(
            get_ready_pool_size=ready_pool,
            get_pool_size=resident_pool,
        )
        manager._max_pool_size = 12
        manager._inflight_samples = 6
        manager._peak_reserved_samples = 0
        manager._capacity_rejections = 0
        manager._peak_inflight_samples = 6
        manager._peak_admitted_ready_samples = 0
        manager._batch_size, manager._engines = 2, SimpleNamespace(max_concurrent=4)

        assert await manager._reserve_pool_capacity(2) is True
        assert manager._inflight_samples == 8
        assert manager._peak_reserved_samples == 12
        assert len(ready_pool.calls) == 1
        assert not resident_pool.calls

    asyncio.run(scenario())


def test_node_local_admission_skips_saturated_node_and_preserves_healthy_progress():
    async def scenario():
        manager = _manager()
        manager._max_pool_size = 8
        manager._batch_size = 1
        manager._engines = EnginePool(
            ["engine-a", "engine-b"],
            engine_nodes=["node-a", "node-b"],
            planned_nodes=["node-a", "node-b"],
        )
        manager._node_ready_watermarks = {"node-a": 4, "node-b": 4}
        manager._pending_tasks = set()
        manager._task_sizes = {}
        manager._task_nodes = {}
        manager._inflight_samples = 0
        manager._inflight_samples_by_node = {"node-a": 0, "node-b": 0}
        manager._peak_inflight_samples = 0
        manager._peak_inflight_samples_by_node = {"node-a": 0, "node-b": 0}
        manager._peak_admitted_ready_samples = 0
        manager._peak_admitted_ready_by_node = {"node-a": 0, "node-b": 0}
        manager._peak_reserved_samples = 0
        manager._peak_reserved_by_node = {"node-a": 0, "node-b": 0}
        manager._capacity_rejections = 0
        manager._capacity_rejections_by_node = {"node-a": 0, "node-b": 0}
        manager._sample_budget = 2
        manager._scheduled_samples = 0
        manager._sample_budget_by_node = {"node-a": 1, "node-b": 1}
        manager._scheduled_samples_by_node = {"node-a": 0, "node-b": 0}
        manager._source_dry = False
        manager._prompt_buffer = deque(
            [
                InferenceInput(
                    data_id="row",
                    prompt="row",
                    metadata=build_work_identity(
                        split="train", epoch=0, source_id="row"
                    ),
                )
            ]
        )

        async def ready_by_node():
            return {"node-a": 4, "node-b": 0}

        manager.controller.get_ready_pool_size_by_node = SimpleNamespace(
            remote=ready_by_node
        )
        observed_nodes = []

        async def dispatch(entries, *, engine_lease=None):
            async with engine_lease as lease:
                observed_nodes.append(lease.node)
                return [(entries[0], ValueError("injected producer failure"))]

        manager._dispatch_batch = dispatch
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)

        assert observed_nodes == ["node-b"]
        assert manager._capacity_rejections_by_node == {"node-a": 1, "node-b": 0}
        assert manager._inflight_samples_by_node == {"node-a": 0, "node-b": 1}
        assert manager._scheduled_samples_by_node == {"node-a": 0, "node-b": 1}
        with pytest.raises(RuntimeError, match="injected producer failure"):
            await manager._collect_completed()
        assert manager._inflight_samples == 0
        assert manager._inflight_samples_by_node == {"node-a": 0, "node-b": 0}
        assert manager._engines.available_by_node == {"node-a": 1, "node-b": 1}
        assert manager._running is False

    asyncio.run(scenario())

def _capacity_manager(ready=0, watermark=8):
    manager = _manager()
    manager._max_pool_size = watermark
    manager._batch_size = 1
    manager._engines = EnginePool(list(range(8)))
    manager._pending_tasks = set()
    manager._task_sizes = {}
    manager._inflight_samples = 0
    manager._peak_inflight_samples = 0
    manager._peak_admitted_ready_samples = 0
    manager._peak_reserved_samples = 0
    manager._capacity_rejections = 0
    manager._sample_budget = 16
    manager._scheduled_samples = 0
    manager._source_dry = False
    manager._prompt_buffer = deque(
        InferenceInput(
            data_id=str(index), prompt=str(index),
            metadata=build_work_identity(split="train", epoch=0, source_id=str(index)),
        )
        for index in range(16)
    )
    pool = SimpleNamespace(ready=ready, forwarded=[])

    async def get_ready():
        return pool.ready

    async def push(results):
        pool.ready += len(results)
        pool.forwarded.extend(results)

    manager.controller.get_ready_pool_size = SimpleNamespace(remote=get_ready)
    manager.controller.push_inference_results = SimpleNamespace(remote=push)
    return manager, pool


@pytest.mark.parametrize("node_count", [2, 4])
def test_released_engine_does_not_reuse_uncollected_node_credit(node_count):
    async def scenario():
        manager, pool = _capacity_manager()
        nodes = [f"node-{index}" for index in range(node_count)]
        manager._engines = EnginePool(
            list(range(node_count)), engine_nodes=nodes, planned_nodes=nodes
        )
        manager._node_ready_watermarks = dict.fromkeys(nodes, 4)
        for field in (
            "_inflight_samples_by_node", "_peak_inflight_samples_by_node",
            "_peak_admitted_ready_by_node", "_peak_reserved_by_node",
            "_capacity_rejections_by_node",
        ):
            setattr(manager, field, dict.fromkeys(nodes, 0))
        manager.controller.get_ready_pool_size_by_node = _AsyncRemote({})
        prior = InferenceInput(
            data_id="prior", prompt="prior",
            metadata=build_work_identity(split="train", epoch=0, source_id="prior"),
        )

        async def completed():
            return [(prior, {"mooncake_key": "key-prior"})]

        # A peer was collected while this engine completed and released its
        # slot. Its output is still pending collection and owns node-0 credit.
        task = asyncio.create_task(completed())
        await asyncio.sleep(0)
        manager._pending_tasks = {task}
        manager._task_sizes = {task: 1}
        manager._task_nodes = {task: nodes[0]}
        manager._inflight_samples = 1
        manager._inflight_samples_by_node[nodes[0]] = 1
        manager._scheduled_samples = 1
        observed = []

        async def dispatch(entries, *, engine_lease=None):
            async with engine_lease as lease:
                observed.append(lease.node)
                return [(entries[0], {"mooncake_key": "key-new"})]

        manager._dispatch_batch = dispatch
        assert manager._engines.available_by_node[nodes[0]] == 1
        assert await manager._start_dispatch(1)
        await asyncio.sleep(0)
        assert observed == [nodes[1]]
        assert manager._inflight_samples_by_node[nodes[0]] == 1
        await manager._collect_completed()
        assert manager._inflight_samples == 0
        assert all(value == 0 for value in manager._inflight_samples_by_node.values())
        assert {row.data_id for row in pool.forwarded} == {"prior", "0"}
        assert manager._engines.available == node_count

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_ready", [0, 7])
def test_finished_engine_refills_until_ready_watermark_without_blocking_collection(
    initial_ready,
):
    async def scenario():
        manager, pool = _capacity_manager(ready=initial_ready)
        gates = [asyncio.Event() for _ in range(16)]

        async def dispatch(entries, *, engine_lease=None):
            async with engine_lease:
                entry = entries[0]
                await gates[int(entry.data_id)].wait()
                return [(entry, {"mooncake_key": "key-" + entry.data_id})]

        manager._dispatch_batch = dispatch
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        assert manager._inflight_samples == 8
        assert manager._engines.available == 0
        gates[0].set()
        await manager._collect_completed()
        assert pool.ready == initial_ready + 1
        assert manager._inflight_samples == 7
        await manager._dispatch_ready_batches()
        if initial_ready == 0:
            # Old ready + in-flight credit rejected this exact replacement.
            assert manager._inflight_samples == 8
            assert manager._scheduled_samples == 9
            assert manager._peak_reserved_samples == 9
        else:
            assert manager._inflight_samples == 7
            assert manager._scheduled_samples == 8
            assert manager._capacity_rejections > 0
        for gate in gates:
            gate.set()
        await manager._drain()
        assert manager._inflight_samples == 0
        assert not manager._pending_tasks and not manager._task_sizes
        assert manager._engines.available == 8
        ids = [item.data_id for item in pool.forwarded]
        assert len(ids) == len(set(ids)) == manager._scheduled_samples
        assert pool.ready <= 8 - 1 + 8
        assert manager._peak_inflight_samples == 8
        assert manager._peak_admitted_ready_samples < 8

    asyncio.run(scenario())


@pytest.mark.parametrize("watermark", [0, 8])
def test_engine_sample_window_remains_bounded_even_without_ready_watermark(watermark):
    async def scenario():
        manager, _pool = _capacity_manager(watermark=watermark)
        manager._inflight_samples = 8
        with pytest.raises(RuntimeError, match="engine dispatch window"):
            await manager._reserve_pool_capacity(1)
        assert manager._inflight_samples == 8
        manager._inflight_samples = 0
        with pytest.raises(RuntimeError, match="engine dispatch window"):
            await manager._reserve_pool_capacity(2)

    asyncio.run(scenario())


def test_collect_failure_preserves_peer_outputs_and_releases_all_inflight_accounting():
    async def scenario():
        manager, pool = _capacity_manager()

        async def dispatch(entries, *, engine_lease=None):
            async with engine_lease:
                entry = entries[0]
                result = (
                    ValueError("failed request")
                    if entry.data_id == "0"
                    else {"mooncake_key": "key-" + entry.data_id}
                )
                return [(entry, result)]

        manager._dispatch_batch = dispatch
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="lost required work"):
            await manager._collect_completed()
        assert manager._inflight_samples == 0
        assert not manager._pending_tasks and not manager._task_sizes
        assert len(pool.forwarded) == 7
        assert manager._running is False

    asyncio.run(scenario())


def test_ready_release_refills_idle_engines_before_the_long_tail_finishes(monkeypatch):
    async def scenario():
        manager, pool = _capacity_manager(ready=7)
        gates = [asyncio.Event() for _ in range(16)]
        initial_started = asyncio.Event()
        tail_waiting = asyncio.Event()
        capacity_released = asyncio.Event()
        refill_forever = asyncio.Event()
        started = []
        real_wait = asyncio.wait

        async def wait(tasks, *, return_when, timeout=None):
            # Advance a virtual admission poll without an elapsed-time sleep.
            if len(tasks) == 1 and not any(task.done() for task in tasks):
                tail_waiting.set()
                if timeout is not None:
                    await capacity_released.wait()
                    return set(), set(tasks)
            return await real_wait(tasks, return_when=return_when, timeout=timeout)

        async def dispatch(entries, *, engine_lease=None):
            async with engine_lease:
                entry = entries[0]
                index = int(entry.data_id)
                started.append(index)
                if len(started) == 8:
                    initial_started.set()
                await gates[index].wait()
                return [(entry, {"mooncake_key": "key-" + entry.data_id})]

        async def refill():
            await refill_forever.wait()

        monkeypatch.setattr(asyncio, "wait", wait)
        manager._dispatch_batch = dispatch
        manager._continuous_refill = refill
        runner = asyncio.create_task(manager.run())
        try:
            await initial_started.wait()
            for gate in gates[:7]:
                gate.set()
            await tail_waiting.wait()
            assert pool.ready == 14
            assert manager._engines.available == 7
            assert manager._scheduled_samples == 8
            pool.ready -= 8  # One consumer step frees the watermark.
            capacity_released.set()
            for _ in range(32):
                await asyncio.sleep(0)
            assert manager._scheduled_samples == 15
            assert not gates[7].is_set()
        finally:
            manager.stop()
            for gate in gates:
                gate.set()
            await runner
        assert manager._inflight_samples == 0
        assert not manager._pending_tasks and not manager._task_sizes
        assert manager._engines.available == 8
        assert len({item.data_id for item in pool.forwarded}) == 15

    asyncio.run(scenario())


def test_admission_poll_preserves_unfinished_work_accounting(monkeypatch):
    async def scenario():
        manager, _pool = _capacity_manager()
        task = asyncio.create_task(asyncio.Event().wait())
        manager._pending_tasks = {task}
        manager._task_sizes[task] = 1
        manager._inflight_samples = 1

        async def poll(tasks, *, return_when, timeout):
            assert tasks == {task} and timeout is not None
            assert return_when == asyncio.FIRST_COMPLETED
            return set(), set(tasks)

        monkeypatch.setattr(asyncio, "wait", poll)
        try:
            await manager._collect_completed()
            assert manager._pending_tasks == {task}
            assert manager._task_sizes == {task: 1}
            assert manager._inflight_samples == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
