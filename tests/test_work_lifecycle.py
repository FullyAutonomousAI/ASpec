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

import pytest

from aspec.controller.inference_manager import has_ready_pool_capacity
from aspec.training.trainer import _consumer_participant
from aspec.work_lifecycle import (
    EXECUTION_ID_KEY,
    WORK_ID_KEY,
    WORK_KIND_KEY,
    WorkLifecycleError,
    WorkLifecycleLedger,
    build_work_identity,
)


def _metadata(source_id: str, *, split: str = "train", epoch: int = 0):
    return build_work_identity(split=split, epoch=epoch, source_id=source_id)


def test_work_identity_is_stable_and_epoch_scoped():
    first = _metadata("row-7", epoch=3)
    assert first == _metadata("row-7", epoch=3)
    assert first != _metadata("row-7", epoch=4)


def test_fenced_discard_releases_only_selected_uncommitted_ownership():
    ledger = WorkLifecycleLedger()
    entries = [_metadata(str(index)) for index in range(2)]
    for index, metadata in enumerate(entries):
        ledger.register_output(data_id=str(index), mooncake_key=str(index),
                               metadata=metadata, size_bytes=100)
    ledger.mark_dispatched(entries[0], expected_consumers=1)
    assert ledger.discard_after_owner_fence([entries[0][EXECUTION_ID_KEY]]) == 1
    assert ledger.resident_outputs == 1
    assert ledger.discard_after_owner_fence([entries[0][EXECUTION_ID_KEY]]) == 0
    assert ledger.resident_bytes == 100


def test_lifecycle_rejects_identity_without_work_kind():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-1")
    metadata.pop(WORK_KIND_KEY)

    with pytest.raises(WorkLifecycleError, match=WORK_KIND_KEY):
        ledger.register_output(
            data_id="row-1",
            mooncake_key="key-1",
            metadata=metadata,
            size_bytes=1,
        )


def test_dispatched_output_remains_resident_until_consumer_release_and_commit():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-1")
    ledger.register_output(
        data_id="row-1",
        mooncake_key="key-1",
        metadata=metadata,
        size_bytes=4096,
    )
    ledger.mark_dispatched(metadata, expected_consumers=1)

    assert ledger.resident_outputs == 1
    assert ledger.resident_bytes == 4096

    ledger.release_output(
        execution_id=metadata[EXECUTION_ID_KEY],
        data_id="row-1",
        mooncake_key="key-1",
        participant="draft_trainer_dp_rank_0",
        outcome="consumed",
    )
    assert ledger.resident_outputs == 0
    assert ledger.status()["state"] == "incomplete"

    ledger.commit_optimizer_step(
        optimizer_step=1,
        execution_ids=[metadata[EXECUTION_ID_KEY]],
        expected_unique=1,
    )
    status = ledger.status()
    assert status["state"] == "validated"
    assert status["committed_work"] == 1
    assert status["optimizer_steps_committed"] == 1
    assert ledger.committed_work_ids() == [metadata[WORK_ID_KEY]]


def test_cancelled_released_work_is_explicitly_discarded():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-cancelled")
    ledger.register_output(
        data_id="row-cancelled",
        mooncake_key="key-cancelled",
        metadata=metadata,
        size_bytes=4096,
    )
    ledger.mark_dispatched(metadata, expected_consumers=1)
    ledger.release_output(
        execution_id=metadata[EXECUTION_ID_KEY],
        data_id="row-cancelled",
        mooncake_key="key-cancelled",
        participant="draft_trainer_dp_rank_0",
        outcome="consumed",
    )

    assert ledger.discard_released_uncommitted() == 1
    status = ledger.status()
    assert status["state"] == "validated"
    assert status["discarded_outputs"] == 1
    assert status["resident_outputs"] == 0
    assert ledger.committed_work_ids() == []


def test_cancel_finalization_does_not_change_committed_work():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-committed")
    ledger.register_output(
        data_id="row-committed",
        mooncake_key="key-committed",
        metadata=metadata,
        size_bytes=1,
    )
    ledger.mark_dispatched(metadata, expected_consumers=1)
    ledger.release_output(
        execution_id=metadata[EXECUTION_ID_KEY],
        data_id="row-committed",
        mooncake_key="key-committed",
        participant="draft_trainer_dp_rank_0",
        outcome="consumed",
    )
    ledger.commit_optimizer_step(
        optimizer_step=1,
        execution_ids=[metadata[EXECUTION_ID_KEY]],
        expected_unique=1,
    )

    assert ledger.discard_released_uncommitted() == 0
    assert ledger.status()["committed_work"] == 1
    assert ledger.committed_work_ids() == [metadata[WORK_ID_KEY]]


def test_duplicate_consumer_release_fails_closed():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-1")
    ledger.register_output(
        data_id="row-1",
        mooncake_key="key-1",
        metadata=metadata,
        size_bytes=1,
    )
    ledger.mark_dispatched(metadata, expected_consumers=2)
    release = {
        "execution_id": metadata[EXECUTION_ID_KEY],
        "data_id": "row-1",
        "mooncake_key": "key-1",
        "participant": "sp-rank-0",
        "outcome": "consumed",
    }
    ledger.release_output(**release)
    with pytest.raises(WorkLifecycleError, match="duplicate release"):
        ledger.release_output(**release)


def test_sequence_shards_have_unique_consumer_identity_and_all_must_release():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-cp2")
    ledger.register_output(
        data_id="row-cp2",
        mooncake_key="key-cp2",
        metadata=metadata,
        size_bytes=1024,
    )
    ledger.mark_dispatched(metadata, expected_consumers=2)

    participants = [
        _consumer_participant("trainer", dp_rank=0, sequence_rank=rank)
        for rank in range(2)
    ]
    assert len(set(participants)) == 2

    for participant in participants:
        ledger.release_output(
            execution_id=metadata[EXECUTION_ID_KEY],
            data_id="row-cp2",
            mooncake_key="key-cp2",
            participant=participant,
            outcome="consumed",
        )

    ledger.commit_optimizer_step(
        optimizer_step=1,
        execution_ids=[metadata[EXECUTION_ID_KEY]] * 2,
        expected_unique=1,
        expected_occurrences=2,
    )
    assert ledger.status()["state"] == "validated"


def test_any_failed_shard_release_keeps_the_output_failed():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-failed-cp2")
    ledger.register_output(
        data_id="row-failed-cp2",
        mooncake_key="key-failed-cp2",
        metadata=metadata,
        size_bytes=1024,
    )
    ledger.mark_dispatched(metadata, expected_consumers=2)

    for participant, outcome in (
        ("cp-rank-0", "discarded_after_error"),
        ("cp-rank-1", "consumed"),
    ):
        ledger.release_output(
            execution_id=metadata[EXECUTION_ID_KEY],
            data_id="row-failed-cp2",
            mooncake_key="key-failed-cp2",
            participant=participant,
            outcome=outcome,
        )

    status = ledger.status()
    assert status["state_counts"] == {"failed": 1}
    assert status["failed_outputs"] == 1


def test_unused_output_is_terminal_only_after_discard_confirmation():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-2")
    ledger.register_output(
        data_id="row-2",
        mooncake_key="key-2",
        metadata=metadata,
        size_bytes=128,
    )
    ledger.begin_discard(metadata)

    assert ledger.status()["state_counts"] == {"discard_pending": 1}
    assert ledger.resident_outputs == 1

    ledger.confirm_discarded([metadata])
    status = ledger.status()
    assert status["state"] == "validated"
    assert status["discarded_outputs"] == 1
    assert status["resident_outputs"] == 0


def test_output_batch_registration_is_atomic_on_duplicate():
    ledger = WorkLifecycleLedger()
    metadata = _metadata("row-3")

    with pytest.raises(WorkLifecycleError, match="duplicate execution_id"):
        ledger.register_outputs(
            [
                {
                    "data_id": "row-3",
                    "mooncake_key": "key-3",
                    "metadata": metadata,
                    "size_bytes": 64,
                },
                {
                    "data_id": "row-3",
                    "mooncake_key": "key-4",
                    "metadata": metadata,
                    "size_bytes": 64,
                },
            ]
        )

    status = ledger.status()
    assert status["produced_outputs"] == 0
    assert status["resident_outputs"] == 0


def test_ready_watermark_is_not_a_pending_or_resident_output_credit():
    assert has_ready_pool_capacity(pool_size=7, max_pool_size=8)
    assert not has_ready_pool_capacity(pool_size=8, max_pool_size=8)
    assert not has_ready_pool_capacity(pool_size=15, max_pool_size=8)
    assert has_ready_pool_capacity(pool_size=100, max_pool_size=0)
    with pytest.raises(ValueError, match="occupancy must be known"):
        has_ready_pool_capacity(pool_size=-1, max_pool_size=8)


def test_failed_attempt_retirement_preserves_commits_and_rejects_late_outputs():
    import threading
    from collections import deque

    from aspec.controller.training_controller import TrainingControllerState
    from aspec.utils.types import InferenceOutput

    controller = TrainingControllerState.__new__(TrainingControllerState)
    controller._pool_lock = threading.Lock()
    controller._eval_pool_lock = threading.Lock()
    controller._lifecycle = WorkLifecycleLedger()
    controller.sample_pool = deque()
    controller.eval_pool = deque()
    controller._sample_bytes = {}
    controller._eval_data_ids = set()
    a, b = _metadata("committed"), _metadata("pending")
    for index, metadata in enumerate((a, b)):
        controller._lifecycle.register_output(data_id=str(index), mooncake_key=str(index),
                                              metadata=metadata, size_bytes=100)
    controller._lifecycle.mark_dispatched(a, expected_consumers=1)
    controller._lifecycle.release_output(execution_id=a[EXECUTION_ID_KEY], data_id="0",
        mooncake_key="0", participant="rank0", outcome="consumed")
    controller._lifecycle.commit_optimizer_step(optimizer_step=1,
        execution_ids=[a[EXECUTION_ID_KEY]], expected_unique=1, expected_occurrences=1)
    with pytest.raises(RuntimeError, match="must be fenced"):
        controller.finalize_fenced_work(owners_fenced=False)
    assert controller._lifecycle.resident_outputs == 1
    result = controller.finalize_fenced_work(owners_fenced=True)
    assert result["discarded"] == 1
    assert result["lifecycle"]["resident_outputs"] == result["lifecycle"]["resident_bytes"] == 0
    assert result["lifecycle"]["committed_work"] == 1
    assert controller.finalize_fenced_work(owners_fenced=True)["discarded"] == 0
    with pytest.raises(RuntimeError, match="after attempt owners"):
        controller.push_inference_results([InferenceOutput(data_id="late", mooncake_key="late", tensor_shapes={},
                                                          metadata=_metadata("late"))])


@pytest.mark.parametrize("kinds", [("train",), ("eval",), ("train", "eval")])
def test_result_admission_is_atomic_with_owner_fencing(kinds):
    """Retire ownership at the first unlocked point after result registration."""
    import threading
    from types import SimpleNamespace

    from aspec.controller.training_controller import TrainingControllerState
    from aspec.utils.types import InferenceOutput

    controller = TrainingControllerState(
        SimpleNamespace(per_dp_rank_batch_size=1),
        dp_size=1,
        with_train_queues=False,
        with_eval_queues=False,
    )
    held = 0
    retired = False

    class RetirementInterleavingLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            nonlocal held
            self.lock.acquire()
            held += 1
            return self

        def __exit__(self, *_exc):
            nonlocal held, retired
            held -= 1
            self.lock.release()
            # Deterministic scheduling: another actor method can run as soon
            # as admission holds neither queue lock. Do not rely on sleeps.
            if held == 0 and not retired and controller._lifecycle.resident_outputs:
                retired = True
                controller.finalize_fenced_work(owners_fenced=True)

    controller._pool_lock = RetirementInterleavingLock()
    controller._eval_pool_lock = RetirementInterleavingLock()
    outputs = []
    for index, kind in enumerate(kinds):
        data_id = f"row-{index}"
        if kind == "eval":
            controller._eval_data_ids.add(data_id)
        outputs.append(InferenceOutput(
            data_id=data_id,
            mooncake_key=f"key-{index}",
            tensor_shapes={"input_ids": (1, 16)},
            metadata=_metadata(data_id, split=kind),
        ))

    controller.push_inference_results(outputs)

    assert retired
    assert not controller.sample_pool
    assert not controller.eval_pool
    assert controller._sample_bytes == {}
    assert controller.get_pool_bytes() == 0
    status = controller.get_work_lifecycle_status()
    assert status["state"] == "validated"
    assert status["discarded_outputs"] == len(outputs)
