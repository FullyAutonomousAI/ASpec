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
# tests/test_capacity_handling.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for Mooncake capacity handling and backpressure mechanism.

Tests cover:
- estimate_tensor_bytes utility function
- AsyncTrainingController pool byte tracking
- Backpressure integration between controller and inference manager
"""

import threading
from dataclasses import dataclass
from queue import Queue
from unittest.mock import patch

import pytest
import torch

from aspec.controller.training_controller import AsyncTrainingController
from aspec.utils.memory import estimate_tensor_bytes
from aspec.work_lifecycle import (
    EXECUTION_ID_KEY,
    WorkLifecycleError,
    build_work_identity,
)


class TestEstimateTensorBytes:
    """Tests for estimate_tensor_bytes utility function."""

    def test_single_tensor_float32(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {"hidden": torch.float32}
        expected = 1 * 128 * 4096 * 4
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_single_tensor_float16(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {"hidden": torch.float16}
        expected = 1 * 128 * 4096 * 2
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_single_tensor_bfloat16(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {"hidden": torch.bfloat16}
        expected = 1 * 128 * 4096 * 2
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_single_tensor_int64(self):
        shapes = {"input_ids": (1, 512)}
        dtypes = {"input_ids": torch.int64}
        expected = 1 * 512 * 8
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_single_tensor_int32(self):
        shapes = {"labels": (1, 512)}
        dtypes = {"labels": torch.int32}
        expected = 1 * 512 * 4
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_single_tensor_bool(self):
        shapes = {"mask": (1, 512)}
        dtypes = {"mask": torch.bool}
        expected = 1 * 512 * 1
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_multiple_tensors(self):
        shapes = {
            "hidden": (1, 128, 4096),
            "logits": (1, 128, 32000),
            "input_ids": (1, 128),
        }
        dtypes = {
            "hidden": torch.bfloat16,
            "logits": torch.float32,
            "input_ids": torch.int64,
        }
        expected = 1 * 128 * 4096 * 2 + 1 * 128 * 32000 * 4 + 1 * 128 * 8
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_missing_dtype_defaults_to_bfloat16(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {}
        expected = 1 * 128 * 4096 * 2
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_string_dtype_float32(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {"hidden": "float32"}
        expected = 1 * 128 * 4096 * 4
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_string_dtype_bfloat16(self):
        shapes = {"hidden": (1, 128, 4096)}
        dtypes = {"hidden": "bfloat16"}
        expected = 1 * 128 * 4096 * 2
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_empty_shapes(self):
        assert estimate_tensor_bytes({}, {}) == 0

    def test_large_tensor(self):
        shapes = {"weights": (8192, 8192)}
        dtypes = {"weights": torch.float32}
        expected = 8192 * 8192 * 4
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_scalar_tensor(self):
        shapes = {"loss": (1,)}
        dtypes = {"loss": torch.float32}
        expected = 1 * 4
        assert estimate_tensor_bytes(shapes, dtypes) == expected

    def test_high_dimensional_tensor(self):
        shapes = {"attention": (2, 32, 128, 128)}
        dtypes = {"attention": torch.float16}
        expected = 2 * 32 * 128 * 128 * 2
        assert estimate_tensor_bytes(shapes, dtypes) == expected


@dataclass
class MockControllerArgs:
    """Mock args for AsyncTrainingController."""

    per_dp_rank_batch_size: int = 4
    max_sample_pool_size: int = 0


def _create_mock_inference_output(
    data_id: str,
    mooncake_key: str,
    tensor_shapes: dict,
    tensor_dtypes: dict,
    work_kind: str = "train",
):
    """Create a mock InferenceOutput."""
    from aspec.utils.types import InferenceOutput

    return InferenceOutput(
        data_id=data_id,
        mooncake_key=mooncake_key,
        tensor_shapes=tensor_shapes,
        tensor_dtypes=tensor_dtypes,
        metadata=build_work_identity(
            split=work_kind,
            epoch=0,
            source_id=data_id,
        ),
    )


def _create_controller_class():
    """Import and return un-decorated AsyncTrainingController class."""
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class

    def create_controller(*args, **kwargs):
        with patch("aspec.controller.training_controller.Queue", Queue):
            return controller_class(*args, **kwargs)

    return create_controller


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_regrouped_accumulation_preserves_cp_release_commit_and_fenced_tail(cp):
    args = MockControllerArgs(per_dp_rank_batch_size=2)
    args.attention_backend = "cudnn_flex_attention"
    args.draft_context_parallel_size = cp
    controller = _create_controller_class()(args, dp_size=2)
    outputs = [
        _create_mock_inference_output(
            str(i), f"key-{i}",
            {"input_ids": (1, length), "hidden_states": (1, length, 8)},
            {"input_ids": torch.int64, "hidden_states": torch.bfloat16},
        )
        for i, length in enumerate([4000, 3900, 200, 100] * 3)
    ]
    controller.push_inference_results(outputs)
    consumed = []
    for microstep in range(2):
        assert controller.try_dispatch_batch(optimizer_step=0, microstep=microstep)
        rows = [[q.get_nowait() for _ in range(2)] for q in controller.train_queues]
        for rank, row in enumerate(rows):
            assert [s.data_id for s in row] == [s.data_id for s in rows[rank // cp * cp]]
            for sample in row:
                assert sample.metadata["optimizer_step"] == 0
                assert sample.metadata["microstep"] == microstep
                consumed.append(sample.metadata[EXECUTION_ID_KEY])
                controller.release_inference_output(
                    execution_id=sample.metadata[EXECUTION_ID_KEY],
                    data_id=sample.data_id, mooncake_key=sample.mooncake_key,
                    participant=f"draft_rank_{rank}", outcome="consumed",
                )
    assert len(consumed) == 8 * cp and len(set(consumed)) == 8
    assert controller.get_ready_pool_size() == 4
    assert controller.get_pool_size() == 4
    committed = controller.commit_optimizer_step(
        optimizer_step=1, execution_ids=consumed, expected_unique=8
    )
    assert committed["committed_work"] == 8
    assert controller.get_status()["batching_dispatched_input_tokens"] == 16400
    assert controller.get_status()["batching_dispatched_context_tokens"] == 16800
    # Unselected future work remains visible to existing owner-fenced cleanup;
    # regrouping must not hide it in a second reservation/cache.
    controller.finalize_fenced_work(owners_fenced=True)
    assert controller.get_pool_size() == 0
    assert controller.get_work_lifecycle_status()["committed_work"] == 8


class TestAsyncTrainingControllerPoolBytes:
    """Tests for pool byte tracking in AsyncTrainingController."""

    def _create_controller(self, per_dp_rank_batch_size: int = 4):
        """Create controller without Ray remote decorator."""
        controller_class = _create_controller_class()
        args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
        with patch("aspec.controller.training_controller.Queue", Queue):
            return controller_class(args, dp_size=1)

    def test_initial_pool_bytes_is_zero(self):
        controller = self._create_controller()
        assert controller.get_pool_bytes() == 0

    def test_push_single_result_updates_bytes(self):
        controller = self._create_controller()

        result = _create_mock_inference_output(
            data_id="test-001",
            mooncake_key="key-001",
            tensor_shapes={"hidden": (1, 128, 4096)},
            tensor_dtypes={"hidden": torch.bfloat16},
        )

        returned_bytes = controller.push_inference_results([result])
        expected_bytes = 1 * 128 * 4096 * 2

        assert returned_bytes == expected_bytes
        assert controller.get_pool_bytes() == expected_bytes

    def test_dispatched_output_releases_queue_credit_but_remains_resident(self):
        controller = self._create_controller(per_dp_rank_batch_size=1)
        result = _create_mock_inference_output(
            data_id="test-001",
            mooncake_key="key-001",
            tensor_shapes={"hidden": (1, 128, 4096)},
            tensor_dtypes={"hidden": torch.bfloat16},
        )
        controller.push_inference_results([result])

        assert controller.get_ready_pool_size() == 1
        assert controller.get_pool_size() == 1
        assert controller.try_dispatch_batch(optimizer_step=0, microstep=0) is True
        assert controller.get_ready_pool_size() == 0
        assert controller.get_pool_size() == 1

    def test_push_multiple_results_accumulates_bytes(self):
        controller = self._create_controller()

        results = [
            _create_mock_inference_output(
                data_id=f"test-{i}",
                mooncake_key=f"key-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(3)
        ]

        returned_bytes = controller.push_inference_results(results)
        expected_bytes = 3 * (1 * 128 * 4096 * 2)

        assert returned_bytes == expected_bytes
        assert controller.get_pool_bytes() == expected_bytes

    def test_push_results_with_different_shapes(self):
        controller = self._create_controller()

        results = [
            _create_mock_inference_output(
                data_id="test-0",
                mooncake_key="key-0",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            ),
            _create_mock_inference_output(
                data_id="test-1",
                mooncake_key="key-1",
                tensor_shapes={"hidden": (1, 256, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            ),
        ]

        returned_bytes = controller.push_inference_results(results)
        expected_bytes = (1 * 128 * 4096 * 2) + (1 * 256 * 4096 * 2)

        assert returned_bytes == expected_bytes

    def test_push_results_with_multiple_tensors_per_sample(self):
        controller = self._create_controller()

        result = _create_mock_inference_output(
            data_id="test-0",
            mooncake_key="key-0",
            tensor_shapes={
                "hidden": (1, 128, 4096),
                "logits": (1, 128, 32000),
            },
            tensor_dtypes={
                "hidden": torch.bfloat16,
                "logits": torch.float32,
            },
        )

        returned_bytes = controller.push_inference_results([result])
        expected_bytes = (1 * 128 * 4096 * 2) + (1 * 128 * 32000 * 4)

        assert returned_bytes == expected_bytes

    def test_push_results_returns_cumulative_bytes(self):
        controller = self._create_controller()

        result1 = _create_mock_inference_output(
            data_id="test-0",
            mooncake_key="key-0",
            tensor_shapes={"hidden": (1, 128, 4096)},
            tensor_dtypes={"hidden": torch.bfloat16},
        )

        result2 = _create_mock_inference_output(
            data_id="test-1",
            mooncake_key="key-1",
            tensor_shapes={"hidden": (1, 256, 4096)},
            tensor_dtypes={"hidden": torch.bfloat16},
        )

        bytes1 = controller.push_inference_results([result1])
        bytes2 = controller.push_inference_results([result2])

        expected_bytes1 = 1 * 128 * 4096 * 2
        expected_bytes2 = expected_bytes1 + (1 * 256 * 4096 * 2)

        assert bytes1 == expected_bytes1
        assert bytes2 == expected_bytes2


class TestAsyncTrainingControllerDispatchDecreasesBytes:
    """Tests that dispatch reduces pool bytes."""

    def _create_controller(self, per_dp_rank_batch_size: int = 2):
        AsyncTrainingController = _create_controller_class()
        args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
        return AsyncTrainingController(args, dp_size=1)

    def test_dispatch_reduces_pool_bytes(self):
        controller = self._create_controller(per_dp_rank_batch_size=2)

        results = [
            _create_mock_inference_output(
                data_id=f"test-{i}",
                mooncake_key=f"key-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(4)
        ]

        controller.push_inference_results(results)
        initial_bytes = controller.get_pool_bytes()
        per_sample_bytes = 1 * 128 * 4096 * 2

        assert initial_bytes == 4 * per_sample_bytes

        dispatched = controller.try_dispatch_batch()
        assert dispatched is True

        assert controller.get_pool_bytes() == 4 * per_sample_bytes
        for _ in range(2):
            sample = controller.train_queues[0].get(timeout=5)
            controller.release_inference_output(
                execution_id=sample.metadata[EXECUTION_ID_KEY],
                data_id=sample.data_id,
                mooncake_key=sample.mooncake_key,
                participant="draft_trainer_dp_rank_0",
                outcome="consumed",
            )
        assert controller.get_pool_bytes() == 2 * per_sample_bytes

    def test_dispatch_removes_correct_bytes_for_varying_sizes(self):
        controller = self._create_controller(per_dp_rank_batch_size=2)

        results = [
            _create_mock_inference_output(
                data_id="test-0",
                mooncake_key="key-0",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            ),
            _create_mock_inference_output(
                data_id="test-1",
                mooncake_key="key-1",
                tensor_shapes={"hidden": (1, 256, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            ),
            _create_mock_inference_output(
                data_id="test-2",
                mooncake_key="key-2",
                tensor_shapes={"hidden": (1, 512, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            ),
        ]

        controller.push_inference_results(results)

        bytes_sample0 = 1 * 128 * 4096 * 2
        bytes_sample1 = 1 * 256 * 4096 * 2
        bytes_sample2 = 1 * 512 * 4096 * 2

        initial_bytes = controller.get_pool_bytes()
        assert initial_bytes == bytes_sample0 + bytes_sample1 + bytes_sample2

        controller.try_dispatch_batch()

        assert controller.get_pool_bytes() == bytes_sample0 + bytes_sample1 + bytes_sample2
        for _ in range(2):
            sample = controller.train_queues[0].get(timeout=5)
            controller.release_inference_output(
                execution_id=sample.metadata[EXECUTION_ID_KEY],
                data_id=sample.data_id,
                mooncake_key=sample.mooncake_key,
                participant="draft_trainer_dp_rank_0",
                outcome="consumed",
            )
        assert controller.get_pool_bytes() == bytes_sample2

    def test_dispatch_all_samples_returns_zero_bytes(self):
        controller = self._create_controller(per_dp_rank_batch_size=2)

        results = [
            _create_mock_inference_output(
                data_id=f"test-{i}",
                mooncake_key=f"key-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(2)
        ]

        controller.push_inference_results(results)
        controller.try_dispatch_batch()

        assert controller.get_pool_bytes() == 2 * 128 * 4096 * 2
        for _ in range(2):
            sample = controller.train_queues[0].get(timeout=5)
            controller.release_inference_output(
                execution_id=sample.metadata[EXECUTION_ID_KEY],
                data_id=sample.data_id,
                mooncake_key=sample.mooncake_key,
                participant="draft_trainer_dp_rank_0",
                outcome="consumed",
            )
        assert controller.get_pool_bytes() == 0

    def test_multiple_dispatches_reduce_bytes_correctly(self):
        controller = self._create_controller(per_dp_rank_batch_size=2)

        results = [
            _create_mock_inference_output(
                data_id=f"test-{i}",
                mooncake_key=f"key-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(6)
        ]

        controller.push_inference_results(results)
        per_sample_bytes = 1 * 128 * 4096 * 2

        assert controller.get_pool_bytes() == 6 * per_sample_bytes

        def dispatch_and_release():
            assert controller.try_dispatch_batch() is True
            for _ in range(2):
                sample = controller.train_queues[0].get(timeout=5)
                controller.release_inference_output(
                    execution_id=sample.metadata[EXECUTION_ID_KEY],
                    data_id=sample.data_id,
                    mooncake_key=sample.mooncake_key,
                    participant="draft_trainer_dp_rank_0",
                    outcome="consumed",
                )

        dispatch_and_release()
        assert controller.get_pool_bytes() == 4 * per_sample_bytes

        dispatch_and_release()
        assert controller.get_pool_bytes() == 2 * per_sample_bytes

        dispatch_and_release()
        assert controller.get_pool_bytes() == 0

    def test_failed_dispatch_does_not_change_bytes(self):
        controller = self._create_controller(per_dp_rank_batch_size=4)

        results = [
            _create_mock_inference_output(
                data_id=f"test-{i}",
                mooncake_key=f"key-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(2)
        ]

        controller.push_inference_results(results)
        initial_bytes = controller.get_pool_bytes()

        dispatched = controller.try_dispatch_batch()
        assert dispatched is False
        assert controller.get_pool_bytes() == initial_bytes


class TestPoolBytesThreadSafety:
    """Tests for thread safety of pool byte tracking."""

    def _create_controller(self, per_dp_rank_batch_size: int = 4):
        AsyncTrainingController = _create_controller_class()
        args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
        return AsyncTrainingController(args, dp_size=1)

    def test_concurrent_pushes_maintain_consistency(self):
        controller = self._create_controller()
        num_threads = 4
        samples_per_thread = 10
        per_sample_bytes = 1 * 128 * 4096 * 2

        def push_samples(thread_id: int):
            for i in range(samples_per_thread):
                result = _create_mock_inference_output(
                    data_id=f"thread-{thread_id}-sample-{i}",
                    mooncake_key=f"key-{thread_id}-{i}",
                    tensor_shapes={"hidden": (1, 128, 4096)},
                    tensor_dtypes={"hidden": torch.bfloat16},
                )
                controller.push_inference_results([result])

        threads = [threading.Thread(target=push_samples, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = num_threads * samples_per_thread * per_sample_bytes
        assert controller.get_pool_bytes() == expected_total
        assert controller.get_pool_size() == num_threads * samples_per_thread


class TestEmptyTensorShapes:
    """Tests for edge cases with empty tensor shapes."""

    def _create_controller(self, per_dp_rank_batch_size: int = 2):
        AsyncTrainingController = _create_controller_class()
        args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
        return AsyncTrainingController(args, dp_size=1)

    def test_push_result_with_empty_shapes(self):
        controller = self._create_controller()

        result = _create_mock_inference_output(
            data_id="test-0",
            mooncake_key="key-0",
            tensor_shapes={},
            tensor_dtypes={},
        )

        returned_bytes = controller.push_inference_results([result])
        assert returned_bytes == 0
        assert controller.get_pool_bytes() == 0
        assert controller.get_pool_size() == 1

    def test_push_result_without_scheduler_identity_fails_closed(self):
        from aspec.utils.types import InferenceOutput

        controller = self._create_controller()
        result = InferenceOutput(
            data_id="test-0", mooncake_key="key-0", tensor_shapes={}
        )

        with pytest.raises(WorkLifecycleError, match="aspec_work_id"):
            controller.push_inference_results([result])

    def test_work_kind_must_match_submitted_scheduler_state(self):
        controller = self._create_controller()
        result = _create_mock_inference_output(
            data_id="not-an-eval-id",
            mooncake_key="key-0",
            tensor_shapes={},
            tensor_dtypes={},
            work_kind="eval",
        )

        with pytest.raises(ValueError, match="disagrees with submitted scheduler state"):
            controller.push_inference_results([result])

    def test_push_result_with_none_shapes(self):
        from aspec.utils.types import InferenceOutput

        controller = self._create_controller()

        result = InferenceOutput(
            data_id="test-0",
            mooncake_key="key-0",
            tensor_shapes=None,
            tensor_dtypes=None,
            metadata=build_work_identity(
                split="train", epoch=0, source_id="test-0"
            ),
        )

        returned_bytes = controller.push_inference_results([result])
        assert returned_bytes == 0
        assert controller.get_pool_size() == 1


class TestEvalDispatch:
    """Tests for eval dispatch pipeline in AsyncTrainingController."""

    def _create_controller(self, per_dp_rank_batch_size=4, max_sample_pool_size=4, dp_size=4):
        AsyncTrainingController = _create_controller_class()
        args = MockControllerArgs(
            per_dp_rank_batch_size=per_dp_rank_batch_size,
            max_sample_pool_size=max_sample_pool_size,
        )
        return AsyncTrainingController(args, dp_size=dp_size)

    def _setup_eval_and_push(self, controller, num_samples):
        """Set up eval dataset, push all inference results, return the eval data_ids."""
        dataset = [{"prompt": f"eval prompt {i}", "data_id": f"s{i}"} for i in range(num_samples)]
        eval_entries = controller._build_eval_entries(dataset)
        controller._eval_expected_count = num_samples
        controller._eval_dispatched_samples = 0
        with controller._prompt_lock:
            controller.prompt_buffer.extendleft(reversed(eval_entries))

        eval_data_ids = sorted(controller._eval_data_ids)
        results = [
            _create_mock_inference_output(
                data_id=data_id,
                mooncake_key=f"mc_{data_id}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
                work_kind="eval",
            )
            for data_id in eval_data_ids
        ]
        controller.push_inference_results(results)
        return eval_data_ids

    def test_get_pool_size_includes_eval(self):
        controller = self._create_controller()

        train_results = [
            _create_mock_inference_output(
                data_id=f"train-{i}",
                mooncake_key=f"key-train-{i}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
            )
            for i in range(2)
        ]
        controller.push_inference_results(train_results)

        self._setup_eval_and_push(controller, 3)

        assert controller.get_pool_size() == 2 + 3
        assert controller.get_ready_pool_size() == 2 + 3

    def test_try_dispatch_eval_batch_happy_path(self):
        controller = self._create_controller(max_sample_pool_size=4)
        self._setup_eval_and_push(controller, 4)

        assert controller.try_dispatch_eval_batch() is True

        from aspec.training.data_fetcher import TrainSample

        for q in controller.eval_queues:
            sample = q.get(timeout=5)
            assert isinstance(sample, TrainSample)

    def test_try_dispatch_eval_batch_insufficient_samples(self):
        controller = self._create_controller(max_sample_pool_size=4)
        self._setup_eval_and_push(controller, 2)

        assert controller.try_dispatch_eval_batch() is False

    def test_finalize_eval_dispatch_all_arrived_and_drained(self):
        controller = self._create_controller(max_sample_pool_size=4, dp_size=4)
        self._setup_eval_and_push(controller, 4)

        assert controller.try_dispatch_eval_batch() is True
        controller.finalize_eval_dispatch()

        assert controller.get_eval_pool_size() == 0
        assert len(controller._eval_data_ids) == 0
        assert controller._eval_expected_count == 0

    def test_finalize_eval_dispatch_fails_when_still_in_flight(self):
        controller = self._create_controller(max_sample_pool_size=4)

        dataset = [{"prompt": f"eval prompt {i}", "data_id": f"s{i}"} for i in range(8)]
        eval_entries = controller._build_eval_entries(dataset)
        controller._eval_expected_count = 8
        controller._eval_dispatched_samples = 0
        with controller._prompt_lock:
            controller.prompt_buffer.extendleft(reversed(eval_entries))

        partial_ids = sorted(controller._eval_data_ids)[:4]
        results = [
            _create_mock_inference_output(
                data_id=data_id,
                mooncake_key=f"mc_{data_id}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
                work_kind="eval",
            )
            for data_id in partial_ids
        ]
        controller.push_inference_results(results)

        assert controller.try_dispatch_eval_batch() is True
        with pytest.raises(RuntimeError, match="before all samples arrived"):
            controller.finalize_eval_dispatch()

    def test_finalize_eval_dispatch_drops_remainder(self):
        controller = self._create_controller(max_sample_pool_size=4, dp_size=4)
        self._setup_eval_and_push(controller, 6)

        controller.try_dispatch_eval_batch()

        assert controller.get_eval_pool_size() == 2
        assert len(controller._eval_data_ids) > 0

        controller.finalize_eval_dispatch()

        assert controller.get_eval_pool_size() == 0
        assert len(controller._eval_data_ids) == 0
        assert controller._eval_expected_count == 0
        assert controller._eval_dispatched_samples == 0

    def test_eval_does_not_contaminate_training_pool(self):
        controller = self._create_controller()
        self._setup_eval_and_push(controller, 4)

        assert len(controller.sample_pool) == 0
        assert len(controller.eval_pool) == 4


class TestEvalEndToEnd:
    """End-to-end eval flow: submit chunks → push results → dispatch batches → finalize."""

    def _create_controller(self, max_sample_pool_size=2, dp_size=2):
        AsyncTrainingController = _create_controller_class()
        args = MockControllerArgs(
            per_dp_rank_batch_size=4,
            max_sample_pool_size=max_sample_pool_size,
        )
        return AsyncTrainingController(args, dp_size=dp_size)

    def test_full_chunked_drain(self):
        controller = self._create_controller(max_sample_pool_size=2)

        num_samples = 7
        dataset = [{"prompt": f"eval prompt {i}", "data_id": f"e{i}"} for i in range(num_samples)]
        eval_entries = controller._build_eval_entries(dataset)
        controller._eval_expected_count = num_samples
        controller._eval_dispatched_samples = 0
        with controller._prompt_lock:
            controller.prompt_buffer.extendleft(reversed(eval_entries))

        eval_data_ids = sorted(controller._eval_data_ids)
        results = [
            _create_mock_inference_output(
                data_id=data_id,
                mooncake_key=f"mc_{data_id}",
                tensor_shapes={"hidden": (1, 128, 4096)},
                tensor_dtypes={"hidden": torch.bfloat16},
                work_kind="eval",
            )
            for data_id in eval_data_ids
        ]
        controller.push_inference_results(results)

        dispatched_count = 0
        while controller.try_dispatch_eval_batch():
            dispatched_count += 1

        assert dispatched_count == 3
        assert controller._eval_dispatched_samples == 6

        assert controller.get_eval_pool_size() == 1

        controller.finalize_eval_dispatch()

        assert controller.get_eval_pool_size() == 0
        assert len(controller._eval_data_ids) == 0
        assert controller._eval_expected_count == 0
        assert controller._eval_dispatched_samples == 0

        from aspec.training.data_fetcher import TrainSample

        for q in controller.eval_queues:
            for _ in range(3):
                sample = q.get(timeout=5)
                assert isinstance(sample, TrainSample)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
