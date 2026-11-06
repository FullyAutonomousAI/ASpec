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

"""Deterministic source-ring scheduling coverage."""

import threading
from collections import deque
from types import SimpleNamespace

import pytest

from aspec.controller.training_controller import (
    AsyncTrainingController,
    take_node_local_balanced_batch,
    take_source_aware_batch,
    take_source_balanced_batch,
)


def _sample(name: str, engine_rank: int | None, request_id: str | None = None):
    metadata = {} if engine_rank is None else {"engine_rank": engine_rank}
    if request_id is not None:
        metadata["request_id"] = request_id
    return SimpleNamespace(data_id=name, metadata=metadata)


def test_source_balanced_batch_preserves_fifo_order_around_selected_sources():
    samples = deque(
        [
            _sample("a", 0),
            _sample("b", 0),
            _sample("c", 1),
            _sample("d", 2),
            _sample("e", 1),
        ]
    )

    selected = take_source_balanced_batch(samples, 3)

    assert [item.data_id for item in selected] == ["a", "c", "d"]
    assert [item.data_id for item in samples] == ["b", "e"]


def test_source_balanced_batch_fills_from_ready_sources_when_diversity_is_missing():
    samples = deque([_sample("a", 0), _sample("b", 0), _sample("c", None)])

    selected = take_source_balanced_batch(samples, 2)

    assert [item.data_id for item in selected] == ["a", "b"]
    assert [item.data_id for item in samples] == ["c"]


def test_source_balanced_batch_waits_only_when_total_samples_are_missing():
    samples = deque([_sample("a", 0), _sample("b", 0)])

    assert take_source_balanced_batch(samples, 3) is None
    assert [item.data_id for item in samples] == ["a", "b"]


def test_source_aware_batch_keeps_complete_producer_request_together():
    samples = deque(
        [
            _sample("a", 0, "request_7_0"),
            _sample("b", 0, "request_7_1"),
            _sample("c", 1, "request_8_0"),
        ]
    )

    selected = take_source_aware_batch(samples, 2)

    assert [item.data_id for item in selected] == ["a", "b"]
    assert [item.data_id for item in samples] == ["c"]


def test_source_aware_batch_falls_back_to_distinct_sources():
    samples = deque(
        [
            _sample("a", 0, "request_7_0"),
            _sample("b", 0, "request_9_0"),
            _sample("c", 1, "request_8_0"),
        ]
    )

    selected = take_source_aware_batch(samples, 2)

    assert [item.data_id for item in selected] == ["a", "c"]
    assert [item.data_id for item in samples] == ["b"]


def test_source_aware_batch_does_not_split_a_larger_producer_request():
    samples = deque(
        [
            _sample("a", 0, "request_7_0"),
            _sample("b", 0, "request_7_1"),
            _sample("c", 0, "request_7_2"),
            _sample("d", 1, "request_8_0"),
        ]
    )

    selected = take_source_aware_batch(samples, 2)

    assert [item.data_id for item in selected] == ["a", "d"]
    assert [item.data_id for item in samples] == ["b", "c"]


def test_source_aware_batch_uses_fifo_when_only_one_source_can_progress():
    samples = deque(
        [
            _sample("a", 0, "request_7_0"),
            _sample("b", 0, "request_9_0"),
            _sample("c", 0, "request_11_0"),
        ]
    )

    selected = take_source_aware_batch(samples, 2)

    assert [item.data_id for item in selected] == ["a", "b"]
    assert [item.data_id for item in samples] == ["c"]


def _node_sample(
    name: str,
    node: str | None,
    engine_rank: int = 0,
    request_id: str | None = None,
):
    metadata = {
        "feature_affinity": "node_local_required",
        "engine_rank": engine_rank,
    }
    if node is not None:
        metadata["source_node_ip"] = node
    if request_id is not None:
        metadata["request_id"] = request_id
    return SimpleNamespace(data_id=name, metadata=metadata)


def test_node_local_batch_balances_sources_inside_each_node():
    samples = deque(
        [
            _node_sample("a0", "10.0.0.1", 0),
            _node_sample("a1", "10.0.0.1", 0),
            _node_sample("b0", "10.0.0.2", 4),
            _node_sample("a2", "10.0.0.1", 1),
            _node_sample("b1", "10.0.0.2", 5),
        ]
    )

    selected = take_node_local_balanced_batch(
        samples, {"10.0.0.1": 2, "10.0.0.2": 2}
    )

    assert [item.data_id for item in selected] == ["a0", "b0", "a2", "b1"]
    assert [item.data_id for item in samples] == ["a1"]


def test_node_local_batch_preserves_complete_requests_inside_each_node():
    samples = deque(
        [
            _node_sample("a0", "10.0.0.1", 0, "request_a_0"),
            _node_sample("b0", "10.0.0.2", 4, "request_b_0"),
            _node_sample("a1", "10.0.0.1", 0, "request_a_1"),
            _node_sample("b1", "10.0.0.2", 4, "request_b_1"),
            _node_sample("a2", "10.0.0.1", 1, "request_c_0"),
            _node_sample("b2", "10.0.0.2", 5, "request_d_0"),
        ]
    )

    selected = take_node_local_balanced_batch(
        samples, {"10.0.0.1": 2, "10.0.0.2": 2}
    )

    assert [item.data_id for item in selected] == ["a0", "b0", "a1", "b1"]
    assert [item.data_id for item in samples] == ["a2", "b2"]


def test_node_local_batch_waits_instead_of_stealing_remote_work():
    samples = deque(
        [
            _node_sample("a0", "10.0.0.1"),
            _node_sample("a1", "10.0.0.1"),
            _node_sample("a2", "10.0.0.1"),
            _node_sample("b0", "10.0.0.2"),
        ]
    )

    assert (
        take_node_local_balanced_batch(
            samples, {"10.0.0.1": 2, "10.0.0.2": 2}
        )
        is None
    )
    assert [item.data_id for item in samples] == ["a0", "a1", "a2", "b0"]


@pytest.mark.parametrize("node", [None, "10.0.0.3"])
def test_node_local_batch_rejects_missing_or_unplanned_source_node(node):
    samples = deque([_node_sample("bad", node)])

    with pytest.raises(ValueError, match="source_node_ip|unplanned"):
        take_node_local_balanced_batch(samples, {"10.0.0.1": 1})


@pytest.mark.parametrize("concurrent_dispatch", [False, True])
def test_node_local_status_reports_ready_supply_and_dispatch_demand(concurrent_dispatch):
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    controller = controller_class.__new__(controller_class)
    controller.prompt_buffer = deque()
    controller.sample_pool = deque(
        [
            _node_sample("a0", "10.0.0.1"),
            _node_sample("a1", "10.0.0.1"),
            _node_sample("b0", "10.0.0.2"),
        ]
    )
    controller._lifecycle = SimpleNamespace(resident_outputs=3, resident_bytes=123)
    controller._pool_lock = threading.Lock()
    controller._eval_pool_lock = threading.Lock()
    controller.eval_pool = deque()
    controller.batch_id = 0
    controller._batching_input_tokens = 0
    controller._batching_context_tokens = 0
    controller._batching_microbatches = 0
    controller.dispatch_batch_size = 4
    controller.feature_affinity = "node_local_required"
    controller._dp_ranks_by_node = {"10.0.0.1": [0, 1], "10.0.0.2": [2, 3]}
    controller.args = SimpleNamespace(per_dp_rank_batch_size=1)
    if concurrent_dispatch:
        class DispatchingPool(deque):
            def __iter__(self):
                iterator = super().__iter__()
                yield next(iterator)
                def dispatch():
                    if controller._pool_lock.acquire(blocking=False):
                        try:
                            self.popleft()
                        finally:
                            controller._pool_lock.release()
                worker = threading.Thread(target=dispatch)
                worker.start()
                worker.join(timeout=5)
                assert not worker.is_alive()
                yield from iterator
        controller.sample_pool = DispatchingPool(controller.sample_pool)

    status = controller.get_status()

    assert status["ready_pool_by_source_node"] == {
        "10.0.0.1": 2,
        "10.0.0.2": 1,
    }
    assert status["dispatch_samples_required_by_node"] == {
        "10.0.0.1": 2,
        "10.0.0.2": 2,
    }
    assert controller.get_ready_pool_size_by_node() == {
        "10.0.0.1": 2,
        "10.0.0.2": 1,
    }


def test_node_local_ready_occupancy_rejects_unplanned_source():
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    controller = controller_class.__new__(controller_class)
    controller.feature_affinity = "node_local_required"
    controller._dp_ranks_by_node = {"10.0.0.1": [0]}
    controller.sample_pool = deque([_node_sample("bad", "10.0.0.2")])
    controller.eval_pool = deque()
    controller._pool_lock = threading.Lock()
    controller._eval_pool_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="unplanned source node"):
        controller.get_ready_pool_size_by_node()
