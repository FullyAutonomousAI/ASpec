# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_partition_results.py. Namespace bindings and ASpec regression coverage adapted.

from dataclasses import dataclass
from queue import Queue
from unittest.mock import patch

import pytest
import torch

from aspec.controller.training_controller import AsyncTrainingController
from aspec.utils.types import InferenceOutput


@dataclass
class MockControllerArgs:
    per_dp_rank_batch_size: int = 2
    max_sample_pool_size: int = 0
    mooncake_feature_affinity: str = "any"
    training_node_ips: list[str] | None = None
    training_num_gpus_per_node: int = 1


def _create_controller_class():
    return AsyncTrainingController.__ray_metadata__.modified_class


def _make_output(
    data_id: str, seq_len: int, *, source_node_ip: str | None = None
) -> InferenceOutput:
    metadata = None
    if source_node_ip is not None:
        metadata = {
            "feature_affinity": "node_local_required",
            "source_node_ip": source_node_ip,
        }
    return InferenceOutput(
        data_id=data_id,
        mooncake_key=f"key-{data_id}",
        tensor_shapes={"input_ids": (1, seq_len), "hidden_states": (1, seq_len, 4096)},
        tensor_dtypes={"input_ids": torch.int64, "hidden_states": torch.bfloat16},
        metadata=metadata,
    )


def _make_controller(dp_size: int, per_dp_rank_batch_size: int):
    controller_class = _create_controller_class()
    args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
    with patch("aspec.controller.training_controller.Queue", Queue):
        return controller_class(args, dp_size=dp_size)


def _make_node_local_controller():
    controller_class = _create_controller_class()
    args = MockControllerArgs(
        per_dp_rank_batch_size=1,
        mooncake_feature_affinity="node_local_required",
        training_node_ips=["10.0.0.1", "10.0.0.2"],
        training_num_gpus_per_node=2,
    )
    with patch("aspec.controller.training_controller.Queue", Queue):
        return controller_class(args, dp_size=4)


class TestPartitionFallback:
    """When at most one sample per rank, partition is round-robin."""

    def test_single_dp_rank_keeps_all_samples_together(self):
        controller = _make_controller(dp_size=1, per_dp_rank_batch_size=4)
        results = [_make_output(f"s{i}", seq_len=100 + i) for i in range(4)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 1
        assert [r.data_id for r in partitions[0]] == ["s0", "s1", "s2", "s3"]

    def test_one_sample_per_rank_uses_round_robin(self):
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=1)
        results = [_make_output(f"s{i}", seq_len=1000 - 100 * i) for i in range(4)]

        partitions = controller._partition_results(results)

        assert [p[0].data_id for p in partitions] == ["s0", "s1", "s2", "s3"]

    def test_empty_results_returns_empty_partitions(self):
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=2)

        partitions = controller._partition_results([])

        assert partitions == [[], [], [], []]

    def test_non_divisible_batch_falls_back_to_round_robin(self):
        # 5 results over 2 ranks: capacity would floor to 2 and the
        # greedy generator would empty out on the 5th item. Fall back
        # to round-robin instead of crashing.
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", seq_len=100 + i) for i in range(5)]

        partitions = controller._partition_results(results)

        assert [r.data_id for r in partitions[0]] == ["s0", "s2", "s4"]
        assert [r.data_id for r in partitions[1]] == ["s1", "s3"]


class TestPartitionMicrobatches:
    """Keep nearby lengths together before assigning complete microbatches."""

    def test_capacity_is_exactly_results_per_rank(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", seq_len=100) for i in range(4)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 2
        assert len(partitions[0]) == 2
        assert len(partitions[1]) == 2

    def test_longest_first_groups_similar_lengths_before_dp_assignment(self):
        # Individual-document LPT mixed long/short rows and padded both ranks
        # to a long sequence. Assign two complete length-adjacent batches.
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            _make_output("a", 1000),
            _make_output("b", 800),
            _make_output("c", 200),
            _make_output("d", 100),
        ]

        partitions = controller._partition_results(results)

        assert [[r.data_id for r in p] for p in partitions] == [["a", "b"], ["c", "d"]]
        assert sum(controller._padded_context_tokens(p) for p in partitions) == 2400

    def test_each_rank_receives_exactly_capacity_samples(self):
        # Stress test: skewed lengths must not violate per-rank capacity.
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=3)
        lengths = [2000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        results = [_make_output(f"s{i}", L) for i, L in enumerate(lengths)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 4
        for p in partitions:
            assert len(p) == 3
        # Every sample is assigned exactly once.
        assigned = sorted(r.data_id for p in partitions for r in p)
        assert assigned == sorted(r.data_id for r in results)

    def test_outlier_does_not_starve_other_ranks(self):
        # One huge sample plus many small ones — the rank holding the
        # outlier should still receive `capacity` samples, not all of them.
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=4)
        results = [_make_output("big", 5000)] + [_make_output(f"s{i}", 100) for i in range(7)]

        partitions = controller._partition_results(results)

        assert len(partitions[0]) == 4
        assert len(partitions[1]) == 4

    def test_partition_is_deterministic_for_fixed_input(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", L) for i, L in enumerate([300, 200, 400, 100])]

        p1 = controller._partition_results(results)
        p2 = controller._partition_results(results)

        ids1 = [[r.data_id for r in part] for part in p1]
        ids2 = [[r.data_id for r in part] for part in p2]
        assert ids1 == ids2


class TestPartitionDefensiveFallback:
    """`_partition_results` should not crash when `input_ids` shape is missing."""

    def test_missing_input_ids_shape_treated_as_zero_length(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            InferenceOutput(
                data_id=f"s{i}",
                mooncake_key=f"k{i}",
                tensor_shapes={"hidden_states": (1, 100, 4096)},  # no "input_ids"
                tensor_dtypes={"hidden_states": torch.bfloat16},
            )
            for i in range(4)
        ]

        partitions = controller._partition_results(results)

        assert len(partitions) == 2
        assert len(partitions[0]) == 2
        assert len(partitions[1]) == 2

    def test_none_tensor_shapes_treated_as_zero_length(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            InferenceOutput(data_id=f"s{i}", mooncake_key=f"k{i}", tensor_shapes=None)
            for i in range(4)
        ]

        partitions = controller._partition_results(results)

        assert sum(len(p) for p in partitions) == 4


class TestNodeLocalPartition:
    def test_partitions_each_node_only_to_its_local_draft_ranks(self):
        controller = _make_node_local_controller()
        results = [
            _make_output("b0", 400, source_node_ip="10.0.0.2"),
            _make_output("a0", 300, source_node_ip="10.0.0.1"),
            _make_output("b1", 200, source_node_ip="10.0.0.2"),
            _make_output("a1", 100, source_node_ip="10.0.0.1"),
        ]

        partitions = controller._partition_results(results)

        assert {
            item.metadata["source_node_ip"] for part in partitions[:2] for item in part
        } == {"10.0.0.1"}
        assert {
            item.metadata["source_node_ip"] for part in partitions[2:] for item in part
        } == {"10.0.0.2"}

    def test_stamp_rejects_cross_node_assignment(self):
        controller = _make_node_local_controller()

        with pytest.raises(RuntimeError, match="affinity violation"):
            controller._stamp_feature_affinity(
                {
                    "feature_affinity": "node_local_required",
                    "source_node_ip": "10.0.0.2",
                },
                dp_rank=0,
                count_dispatch=True,
            )


def _dispatch_ids(controller, results, step, microstep):
    controller._dispatch_to_queues(
        results, controller.train_queues, optimizer_step=step, microstep=microstep
    )
    return [[queue.get_nowait().data_id for _ in range(controller.args.per_dp_rank_batch_size)]
            for queue in controller.train_queues]


def test_complete_microbatches_balance_across_accumulation_not_inside_batches():
    controller = _make_controller(2, 2)
    first = [_make_output(f"a{i}", n) for i, n in enumerate([4000, 3900, 200, 100])]
    second = [_make_output(f"b{i}", n) for i, n in enumerate([4000, 3900, 200, 100])]
    assert _dispatch_ids(controller, first, 7, 0) == [["a0", "a1"], ["a2", "a3"]]
    assert _dispatch_ids(controller, second, 7, 1) == [["b2", "b3"], ["b0", "b1"]]
    assert controller._partition_rank_loads == [8400, 8400]
    status = controller.get_status()
    assert status["batching_dispatched_input_tokens"] == 16400
    assert status["batching_dispatched_context_tokens"] == 16800
    assert status["batching_dispatched_microbatches"] == 4


def test_optimizer_boundary_and_fresh_resume_reset_partition_loads():
    controller = _make_controller(2, 2)
    results = [_make_output(str(i), n) for i, n in enumerate([4000, 3900, 200, 100])]
    _dispatch_ids(controller, results, 7, 0)
    # A new step reconstructs all scheduling loads, just as a fresh resumed
    # controller does. Nothing beyond committed optimizer state is required.
    expected = _dispatch_ids(_make_controller(2, 2), results, 8, 0)
    assert _dispatch_ids(controller, results, 8, 0) == expected
    assert controller._partition_rank_loads == [8000, 400]


def test_eval_does_not_mutate_training_loads_or_dispatched_shape_counters():
    controller = _make_controller(2, 2)
    results = [_make_output(str(i), n) for i, n in enumerate([4000, 3900, 200, 100])]
    _dispatch_ids(controller, results, 7, 0)
    before = list(controller._partition_rank_loads), controller.get_status()
    controller._dispatch_to_queues(results, controller.eval_queues)
    assert controller._partition_rank_loads == before[0]
    assert controller.get_status() == before[1]
    assert _dispatch_ids(controller, results, 7, 1) == [["2", "3"], ["0", "1"]]


def test_node_local_groups_balance_only_over_their_local_cp_replicas():
    cls = _create_controller_class()
    args = MockControllerArgs(per_dp_rank_batch_size=2,
                              mooncake_feature_affinity="node_local_required",
                              training_node_ips=["10.0.0.1", "10.0.0.2"],
                              training_num_gpus_per_node=4)
    args.attention_backend = "cudnn_flex_attention"
    args.draft_context_parallel_size = 2
    with patch("aspec.controller.training_controller.Queue", Queue):
        controller = cls(args, dp_size=4)
    results = [_make_output(f"{node}-{i}", n, source_node_ip=node)
               for node in args.training_node_ips
               for i, n in enumerate([4000, 3900, 200, 100])]
    controller._dispatch_to_queues(results, controller.train_queues, optimizer_step=0, microstep=0)
    rows = [[queue.get_nowait() for _ in range(2)] for queue in controller.train_queues]
    assert [[s.data_id for s in row] for row in rows[::2]] == [
        ["10.0.0.1-0", "10.0.0.1-1"], ["10.0.0.1-2", "10.0.0.1-3"],
        ["10.0.0.2-0", "10.0.0.2-1"], ["10.0.0.2-2", "10.0.0.2-3"],
    ]
    for rank in range(0, 8, 2):
        assert [s.data_id for s in rows[rank]] == [s.data_id for s in rows[rank + 1]]
        assert all(s.metadata["assigned_draft_node_ip"] == ("10.0.0.1" if rank < 4 else "10.0.0.2")
                   for s in rows[rank])
    assert controller.get_status()["batching_dispatched_input_tokens"] == 16400
    assert controller.get_status()["batching_dispatched_context_tokens"] == 16800


def test_stateless_partition_does_not_mutate_callers_loads():
    controller = _make_controller(2, 2)
    loads = [10000, 0]
    results = [_make_output(str(i), n) for i, n in enumerate([4000, 3900, 200, 100])]
    partitions = controller._partition_result_subset(results, 2, rank_loads=loads)
    assert [[r.data_id for r in p] for p in partitions] == [["2", "3"], ["0", "1"]]
    assert loads == [10000, 0]
