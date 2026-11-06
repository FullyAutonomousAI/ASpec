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

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aspec.controller import shared_source
from aspec.transfer.mooncake.p2p import (
    GenerationSlotRing,
    MooncakeP2PDescriptor,
    MooncakeP2PTensorDescriptor,
)
from aspec.utils.types import InferenceInput, InferenceOutput
from aspec.work_lifecycle import EXECUTION_ID_KEY, WorkLifecycleLedger, build_work_identity


def test_shared_slot_waits_for_every_reader_and_late_ack_does_not_free_next_generation():
    ring = GenerationSlotRing(1)
    lease = ring.acquire("sample")
    ring.register_readers(lease, ["a", "b"])
    assert ring.release_reader(lease, "a") is False
    assert ring.release_reader(lease, "a") is False
    with pytest.raises(TimeoutError):
        ring.acquire("next", timeout=0)
    with pytest.raises(RuntimeError, match="pending readers"):
        ring.release_acknowledged(lease.slot_id, lease.generation, lease.owner)
    with pytest.raises(RuntimeError, match="unknown"):
        ring.release_reader(lease, "foreign")
    assert ring.release_reader(lease, "b") is True
    next_lease = ring.acquire("next")
    assert ring.release_reader(lease, "a") is False
    assert ring.in_use == 1
    ring.release(next_lease.slot_id, next_lease.generation, next_lease.owner)


def test_audience_is_frozen_and_repeated_registration_does_not_restore_a_read():
    ring = GenerationSlotRing(1)
    lease = ring.acquire("sample")
    ring.register_readers(lease, ["a", "b"])
    ring.release_reader(lease, "a")
    ring.register_readers(lease, ["a", "b"])
    assert not ring.reader_pending(lease, "a")
    with pytest.raises(RuntimeError, match="cannot change"):
        ring.register_readers(lease, ["b"])
    ring.release_reader(lease, "b")
    ring.acquire("sample")
    with pytest.raises(RuntimeError, match="stale"):
        ring.register_readers(lease, ["a", "b"])


class _Controller:
    def __init__(self, args, _dp, *, with_train_queues=True, with_eval_queues=True):
        self.args = copy(args)
        self.args.per_dp_rank_batch_size = 1
        self.train_queues, self.eval_queues = [], []
        self.queue_roles = (with_train_queues, with_eval_queues)
        self._dataset_seed = 42
        self._source_balanced_dispatch = True
        self._dataset_epoch = 0
        self._restored_committed_work_ids = set()
        self.prompt_buffer = deque()
        self.outputs = []
        self.sample_pool = deque()
        self._sample_bytes = {}
        self._lifecycle = WorkLifecycleLedger()

    def load_dataset(self, _args):
        self._stored_dataset = ["0", "1", "2", "3"]
        self._dataset_fingerprint = "same-input"
        self._dataset_admission = {}
        return 4

    def submit_training_dataset(self, **kwargs):
        self._dataset_epoch = kwargs.get("epoch", 0)
        self._restored_committed_work_ids = set(kwargs.get("committed_work_ids", []))
        self.prompt_buffer = deque(
            InferenceInput(
                data_id=value,
                metadata=build_work_identity(
                    split="train", epoch=self._dataset_epoch, source_id=value
                ),
            )
            for value in self._stored_dataset
        )
        return 4

    def get_checkpoint_work_state(self):
        return {"dataset_epoch": self._dataset_epoch}

    def load_eval_dataset(self, _args):
        self._stored_eval_dataset = []
        self._eval_dataset_admission = {"admitted_samples": 0}
        return 0

    def get_prompts(self, count):
        return [self.prompt_buffer.popleft() for _ in range(min(count, len(self.prompt_buffer)))]

    def reload_dataset(self):
        self._dataset_epoch += 1
        return self.submit_training_dataset(epoch=self._dataset_epoch)

    def push_inference_results(self, outputs):
        self.outputs.extend(outputs)
        self.sample_pool.extend(outputs)
        for output in outputs:
            self._lifecycle.register_output(
                data_id=output.data_id,
                mooncake_key=output.mooncake_key,
                metadata=output.metadata,
                size_bytes=2,
            )

    def finalize_uncommitted_work(self):
        return self._lifecycle.discard_released_uncommitted()

    def set_inference_error(self, message):
        self._inference_error = message

    def get_ready_pool_size(self):
        return len(self.outputs)


def test_shared_eval_fanout_does_not_spend_training_demand(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=4, global_batch_size=2, eval_data_path="eval")
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.source._stored_eval_dataset = list(range(6))
    hub.source._eval_data_ids = set()
    for name, count in (("a", 6), ("b", 3)):
        hub.experiments[name]._stored_eval_dataset = list(range(count))
        hub.experiments[name]._eval_data_ids = set()
    hub.source._build_eval_entries = lambda values: [InferenceInput(
        data_id=f"eval-{value}", metadata=build_work_identity(split="eval", epoch=0, source_id=str(value))
    ) for value in values]
    before = dict(hub.remaining)
    assert hub.begin_eval(["a", "b"]) == 6
    entries = hub.get_prompts(4)
    assert len(entries) == 4
    assert hub.remaining == before
    assert hub._audiences[entries[0].metadata[EXECUTION_ID_KEY]] == ("a", "b")
    assert hub._audiences[entries[3].metadata[EXECUTION_ID_KEY]] == ("a",)
    assert len(hub.experiments["b"]._eval_data_ids) == 3
    with pytest.raises(RuntimeError, match="still owns"):
        hub.end_eval()


def _output(entry, slot):
    descriptor = MooncakeP2PDescriptor(
        version=3,
        run_id="run",
        attempt_id="attempt",
        store_key=f"key-{slot}",
        data_id=entry.data_id,
        request_id="request",
        source_engine="0",
        source_endpoint="localhost:1",
        source_release_endpoint="localhost:2",
        slot_base_ptr=1024,
        slot_id=slot,
        generation=1,
        payload_bytes=2,
        tensors=(MooncakeP2PTensorDescriptor("hidden_states", (1,), "bfloat16", 0, 2),),
    )
    return InferenceOutput(
        entry.data_id, descriptor.to_key(), {"hidden_states": (1,)}, metadata=entry.metadata
    )


def test_fanout_preserves_source_order_independent_horizons_and_safe_retirement(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    controls = []
    monkeypatch.setattr(
        shared_source, "source_control", lambda descriptor, **op: controls.append((descriptor, op))
    )
    args = SimpleNamespace(dp_size=1, num_train_steps=2, global_batch_size=1)
    longer = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": longer})
    hub.load_dataset(args)
    assert hub.experiments["a"]._eval_dataset_admission == {"admitted_samples": 0}
    assert hub.get_prompts(1) == []  # do not race experiment subscription setup
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    entries = hub.get_prompts(3)
    hub.push_inference_results([_output(entries[1], 1)])
    assert hub.experiments["a"].outputs == []
    hub.push_inference_results([_output(entries[0], 0), _output(entries[2], 2)])
    assert [out.data_id for out in hub.experiments["a"].outputs] == ["0", "1"]
    assert [out.data_id for out in hub.experiments["b"].outputs] == ["0", "1", "2"]
    assert [op["readers"] for _, op in controls] == [["a", "b"], ["a", "b"], ["b"]]
    assert hub.get_ready_pool_size() == 3
    with pytest.raises(RuntimeError, match="owners"):
        hub.retire("a", owners_fenced=False)
    hub.retire("a", owners_fenced=True)
    assert hub.status()["active_experiments"] == ["b"]
    assert all(op == {"operation": "withdraw", "reader_id": "a"} for _, op in controls[-2:])
    assert hub.status()["published_source_observations"] == 3
    assert hub.get_pool_size() == 3
    assert hub.get_prompts(1) == []


def test_retired_experiment_is_not_readded_by_inflight_completion(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    controls = []
    monkeypatch.setattr(
        shared_source, "source_control", lambda descriptor, **op: controls.append(op)
    )
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    entry = hub.get_prompts(1)[0]
    hub.retire("a", owners_fenced=True)
    hub.push_inference_results([_output(entry, 0)])
    assert controls == [{"operation": "subscribe", "readers": ["b"]}]
    assert hub.experiments["a"].outputs == []
    assert (
        hub.experiments["b"].outputs[0].metadata[EXECUTION_ID_KEY]
        == entry.metadata[EXECUTION_ID_KEY]
    )


def test_completed_reorder_backlog_counts_for_credit_and_storage_before_publication(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    monkeypatch.setattr(shared_source, "source_control", lambda *_a, **_k: None)
    args = SimpleNamespace(dp_size=1, num_train_steps=4, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    entries = hub.get_prompts(4)
    hub.push_inference_results([_output(entries[index], index) for index in (1, 2, 3)])
    assert hub.experiments["a"].get_ready_pool_size() == 0
    assert hub.get_ready_pool_size() == 3
    assert hub.get_pool_size() == 3
    assert hub.get_pool_bytes() == 6
    assert hub.status()["completed_unpublished_observations"] == 3
    assert hub.status()["peak_outstanding_payload_bytes"] == 6
    hub.push_inference_results([_output(entries[0], 0)])
    assert hub.get_ready_pool_size() == 4
    assert hub.get_pool_size() == 4  # One storage allocation, not two reader views.
    assert hub.get_pool_bytes() == 8
    assert hub.status()["completed_unpublished_observations"] == 0
    for name in ("a", "b"):
        hub.retire(name, owners_fenced=True)
    assert hub.finish_production(inflight_samples=0)["outstanding_payload_bytes"] == 0


def test_cancellation_stops_only_new_demand_then_fenced_retirement_reclaims_work(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    controls = []
    monkeypatch.setattr(
        shared_source, "source_control", lambda descriptor, **op: controls.append(op)
    )
    args = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    first = hub.get_prompts(1)[0]
    hub.push_inference_results([_output(first, 0)])
    hub.request_cancel("a", "test")
    with pytest.raises(RuntimeError, match="cancelled"):
        hub.invoke("a", "try_dispatch_batch", (), {})
    # A still holds its published read right until the process-level fence.
    assert hub.experiments["a"]._lifecycle.resident_outputs == 1
    hub.retire("a", owners_fenced=True)
    assert hub.experiments["a"]._lifecycle.resident_outputs == 0
    second = hub.get_prompts(1)[0]
    hub.push_inference_results([_output(second, 1)])
    assert controls[-1] == {"operation": "subscribe", "readers": ["b"]}
    assert hub.remaining["b"] == 1


def test_restored_work_is_skipped_per_experiment_not_per_shared_stream(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    controls = []
    monkeypatch.setattr(
        shared_source, "source_control", lambda descriptor, **op: controls.append(op)
    )
    args = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    restored = build_work_identity(split="train", epoch=0, source_id="0")["aspec_work_id"]
    hub.invoke(
        "a", "submit_training_dataset", (), {"committed_work_ids": [restored], "completed_steps": 1}
    )
    hub.invoke("b", "submit_training_dataset", (), {})
    entries = hub.get_prompts(3)
    hub.push_inference_results([_output(entry, index) for index, entry in enumerate(entries)])
    assert [output.data_id for output in hub.experiments["a"].outputs] == ["1", "2"]
    assert [output.data_id for output in hub.experiments["b"].outputs] == ["0", "1", "2"]
    assert controls[0] == {"operation": "subscribe", "readers": ["b"]}


@pytest.mark.parametrize("method", ["try_dispatch_batch", "try_dispatch_eval_batch"])
def test_dispatch_round_trip_does_not_serialize_independent_experiments(monkeypatch, method):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    entered, release = Event(), Event()

    def blocked_dispatch():
        entered.set()
        assert release.wait(timeout=1)
        return True

    setattr(hub.experiments["a"], method, blocked_dispatch)
    setattr(hub.experiments["b"], method, lambda: True)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(hub.invoke, "a", method, (), {})
        try:
            assert entered.wait(timeout=1)
            second = executor.submit(hub.invoke, "b", method, (), {})
            assert second.result(timeout=1)
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=1)


def test_shared_source_failure_is_broadcast_without_new_admission(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.set_inference_error("producer failed")
    with pytest.raises(RuntimeError, match="producer failed"):
        hub.get_prompts(1)
    assert all(child._inference_error == "producer failed" for child in hub.experiments.values())


def test_abandoned_admission_is_not_reported_as_resident_source_storage(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    monkeypatch.setattr(shared_source, "source_control", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(dp_size=1, num_train_steps=2, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    hub.get_prompts(2)
    with pytest.raises(RuntimeError, match="active"):
        hub.finish_production(inflight_samples=0)
    for name in ("a", "b"):
        hub.retire(name, owners_fenced=True)
    status = hub.finish_production(inflight_samples=0)
    assert status["pending_production"] == 0
    assert status["abandoned_admissions_without_output"] == 2
    assert status["outstanding_payload_bytes"] == 0


def test_resume_starts_at_oldest_checkpoint_epoch_not_epoch_zero(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    monkeypatch.setattr(shared_source, "source_control", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(dp_size=1, num_train_steps=2, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    hub.invoke("a", "submit_training_dataset", (), {"epoch": 2})
    hub.invoke("b", "submit_training_dataset", (), {"epoch": 3})
    assert hub.source._dataset_epoch == 2
    assert hub.invoke("b", "get_checkpoint_work_state", (), {})["dataset_epoch"] == 3
    entries = hub.get_prompts(4)
    assert [entry.metadata["shared_source_epoch"] for entry in entries] == [2, 2, 3, 3]
    hub.push_inference_results([_output(entry, i) for i, entry in enumerate(entries)])
    assert [out.metadata["shared_source_epoch"] for out in hub.experiments["a"].outputs] == [2, 2]
    assert [out.metadata["shared_source_epoch"] for out in hub.experiments["b"].outputs] == [3, 3]


def test_duplicate_subscription_does_not_subtract_resume_budget_twice(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args})
    hub.load_dataset(args)
    kwargs = {"epoch": 1, "committed_work_ids": ["restored"], "completed_steps": 1}
    assert hub.invoke("a", "submit_training_dataset", (), kwargs) == 4
    assert hub.invoke("a", "submit_training_dataset", (), kwargs) == 4
    assert hub.remaining["a"] == 2
    with pytest.raises(RuntimeError, match="cannot change"):
        hub.invoke("a", "submit_training_dataset", (), {"epoch": 2})


def test_cp_view_preparation_failure_does_not_create_unregistered_read_rights(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    control = Mock()
    monkeypatch.setattr(shared_source, "source_control", control)
    original = shared_source.shared_source_cp_views

    def prepare(descriptor, name, *args, **kwargs):
        if name == "b":
            raise ValueError("invalid CP feature view")
        return original(descriptor, name, *args, **kwargs)

    monkeypatch.setattr(shared_source, "shared_source_cp_views", prepare)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    output = _output(hub.get_prompts(1)[0], 0)
    with pytest.raises(ValueError, match="invalid CP feature"):
        hub.push_inference_results([output])
    assert hub._descriptors == {"a": {}, "b": {}}
    assert hub._pending_subscriptions == hub._reader_payloads == {}
    control.assert_not_called()
    for name in ("a", "b"):
        hub.retire(name, owners_fenced=True)
    status = hub.finish_production(inflight_samples=0)
    control.assert_called_once_with(MooncakeP2PDescriptor.from_key(output.mooncake_key))
    assert status["discarded_unpublished_observations"] == 1
    assert status["outstanding_payload_bytes"] == status["pending_production"] == 0


def test_partial_fanout_failure_keeps_all_registered_read_rights(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    controls = []
    monkeypatch.setattr(
        shared_source, "source_control", lambda descriptor, **op: controls.append(op)
    )
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    hub.experiments["a"].push_inference_results = Mock(side_effect=RuntimeError("fanout failed"))
    with pytest.raises(RuntimeError, match="fanout failed"):
        hub.push_inference_results([_output(hub.get_prompts(1)[0], 0)])
    for name in ("a", "b"):
        hub.retire(name, owners_fenced=True)
    assert controls == [
        {"operation": "subscribe", "readers": ["a", "b"]},
        {"operation": "withdraw", "reader_id": "a"},
        {"operation": "withdraw", "reader_id": "b"},
    ]


def test_retirement_waits_for_own_dispatch_without_blocking_other_experiment(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    entered, release = Event(), Event()
    queues = [Mock(), Mock()]
    hub.experiments["a"].train_queues = queues[:1]
    hub.experiments["a"].eval_queues = queues[1:]

    def blocked_dispatch():
        entered.set()
        assert release.wait(timeout=2)
        return True

    hub.experiments["a"].try_dispatch_batch = blocked_dispatch
    hub.experiments["b"].try_dispatch_batch = lambda: True
    with ThreadPoolExecutor(3) as executor:
        dispatch = executor.submit(hub.invoke, "a", "try_dispatch_batch", (), {})
        try:
            assert entered.wait(timeout=1)
            retire = executor.submit(hub.retire, "a", owners_fenced=True)
            assert executor.submit(hub.invoke, "b", "try_dispatch_batch", (), {}).result(timeout=1)
            assert not retire.done()
        finally:
            release.set()
        assert dispatch.result(timeout=1)
        retire.result(timeout=1)
    for queue in queues:
        queue.shutdown.assert_called_once_with(force=True)
    with pytest.raises(RuntimeError, match="retired"):
        hub.invoke("a", "try_dispatch_batch", (), {})


def test_running_check_handles_cancel_even_when_next_step_is_already_prefetched(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.request_cancel("a", "requested during native compute")
    with pytest.raises(RuntimeError, match="cancelled"):
        hub.invoke("a", "check_running", (), {})
    assert hub.invoke("b", "check_running", (), {}) is None
    hub.set_inference_error("source failed")
    with pytest.raises(RuntimeError, match="source failed"):
        hub.invoke("b", "check_running", (), {})


def test_failed_reader_withdrawal_can_be_retried_without_false_retirement(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    control = Mock()
    monkeypatch.setattr(shared_source, "source_control", control)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args})
    hub.load_dataset(args)
    hub.invoke("a", "submit_training_dataset", (), {})
    hub.push_inference_results([_output(hub.get_prompts(1)[0], 0)])
    control.side_effect = RuntimeError("withdrawal unavailable")
    with pytest.raises(RuntimeError, match="withdrawal unavailable"):
        hub.retire("a", owners_fenced=True)
    assert hub.status()["active_experiments"] == []
    assert hub.status()["outstanding_observations"] == 1
    assert hub.status()["retiring_experiments"] == ["a"]
    assert hub._retired == set()
    assert hub.experiments["a"]._lifecycle.resident_outputs == 1
    control.side_effect = None
    hub.retire("a", owners_fenced=True)
    assert hub._retired == {"a"}
    assert hub.experiments["a"]._lifecycle.resident_outputs == 0


def test_dataset_loaded_once_resolves_each_auto_horizon_without_unused_queue_actors(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    automatic = SimpleNamespace(dp_size=1, num_train_steps=None, global_batch_size=1, num_epochs=2)
    explicit = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(automatic, {"a": automatic, "b": explicit})
    assert hub.load_dataset(automatic) == 4
    assert hub.remaining == {"a": 8, "b": 3}
    assert hub.source.queue_roles == (False, False)
    assert all(child.queue_roles == (True, False) for child in hub.experiments.values())


@pytest.mark.parametrize("completed_steps,remaining", [(16, 32), (20, 0)])
def test_resume_budget_uses_remaining_steps_not_old_global_batch_size(
    monkeypatch, completed_steps, remaining
):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=20, global_batch_size=8)
    hub = shared_source.SharedSourceState(args, {"a": args})
    hub.load_dataset(args)
    hub.invoke(
        "a",
        "submit_training_dataset",
        (),
        {
            "completed_steps": completed_steps,
            "committed_work_ids": [f"past-{i}" for i in range(completed_steps * 4)],
        },
    )
    assert hub.remaining["a"] == remaining  # Old work used GBS4; this attempt uses GBS8.
    if not remaining:
        assert hub.get_prompts(1) == []

@pytest.mark.parametrize("ack_lost", [False, True])
def test_unconfirmed_source_subscription_survives_failure_and_fenced_retry(monkeypatch, ack_lost):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    ring = GenerationSlotRing(1)
    lease = ring.acquire("key-0")
    unavailable = True
    controls = []

    def control(descriptor, **operation):
        controls.append(operation)
        if operation.get("operation") == "subscribe":
            if not unavailable or ack_lost:
                ring.register_readers(lease, operation["readers"])
            if unavailable:
                raise RuntimeError("subscription ACK unavailable")
        elif operation.get("operation") == "withdraw":
            ring.release_reader(lease, operation["reader_id"])
        else:
            ring.release_acknowledged(lease.slot_id, lease.generation, lease.owner)

    monkeypatch.setattr(shared_source, "source_control", control)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args, "b": args})
    hub.load_dataset(args)
    for name in ("a", "b"):
        hub.invoke(name, "submit_training_dataset", (), {})
    with pytest.raises(RuntimeError, match="ACK unavailable"):
        hub.push_inference_results([_output(hub.get_prompts(1)[0], 0)])
    assert hub.status()["outstanding_observations"] == 1
    assert hub.status()["outstanding_payload_bytes"] == 2
    assert hub.status()["completed_unpublished_observations"] == 1
    assert hub.status()["pending_source_subscriptions"] == 1
    assert all(not child.outputs for child in hub.experiments.values())
    with pytest.raises(RuntimeError, match="owners"):
        hub.retire("a", owners_fenced=False)
    with pytest.raises(RuntimeError, match="ACK unavailable"):
        hub.retire("a", owners_fenced=True)
    assert hub.status()["retiring_experiments"] == ["a"]
    assert hub.status()["outstanding_observations"] == 1
    assert ring.in_use == 1

    unavailable = False
    hub.retire("a", owners_fenced=True)
    assert ring.reader_pending(lease, "b")
    assert ring.in_use == 1
    hub.retire("b", owners_fenced=True)
    assert ring.in_use == 0
    next_lease = ring.acquire("next-generation")
    status = hub.finish_production(inflight_samples=0)
    assert status["outstanding_observations"] == 0
    assert status["pending_source_subscriptions"] == 0
    assert status["discarded_unpublished_observations"] == 1
    assert status["abandoned_admissions_without_output"] == 0
    assert ring.in_use == 1  # late cleanup of the old generation must not release this one
    ring.release(next_lease.slot_id, next_lease.generation, next_lease.owner)
    assert all(op["readers"] == ["a", "b"] for op in controls if op.get("operation") == "subscribe")


def test_unsubscribed_source_release_failure_remains_visible_until_retry(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args})
    hub.load_dataset(args)
    hub.invoke("a", "submit_training_dataset", (), {})
    output = _output(hub.get_prompts(1)[0], 0)
    hub.retire("a", owners_fenced=True)
    control = Mock(side_effect=RuntimeError("release unavailable"))
    monkeypatch.setattr(shared_source, "source_control", control)
    with pytest.raises(RuntimeError, match="release unavailable"):
        hub.push_inference_results([output])
    assert hub.status()["outstanding_observations"] == 1
    with pytest.raises(RuntimeError, match="release unavailable"):
        hub.finish_production(inflight_samples=0)
    assert hub.status()["outstanding_observations"] == 1
    control.side_effect = None
    status = hub.finish_production(inflight_samples=0)
    assert status["outstanding_observations"] == 0
    assert status["discarded_unpublished_observations"] == 1
    assert status["abandoned_admissions_without_output"] == 0

def test_partial_unpublished_source_cleanup_is_retryable_without_double_accounting(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=3, global_batch_size=1)
    hub = shared_source.SharedSourceState(args, {"a": args})
    hub.load_dataset(args)
    hub.invoke("a", "submit_training_dataset", (), {})
    entries = hub.get_prompts(3)
    # Two completed outputs wait behind one missing prefix; none was subscribed.
    hub.push_inference_results([_output(entries[1], 1), _output(entries[2], 2)])
    hub.retire("a", owners_fenced=True)
    control = Mock(side_effect=[None, RuntimeError("release unavailable")])
    monkeypatch.setattr(shared_source, "source_control", control)
    with pytest.raises(RuntimeError, match="release unavailable"):
        hub.finish_production(inflight_samples=0)
    status = hub.status()
    assert status["outstanding_observations"] == 1
    assert status["completed_unpublished_observations"] == 1
    assert status["discarded_unpublished_observations"] == 1
    control.side_effect = None
    status = hub.finish_production(inflight_samples=0)
    assert status["outstanding_observations"] == 0
    assert status["discarded_unpublished_observations"] == 2
    assert status["abandoned_admissions_without_output"] == 1

def test_packed_horizon_is_rows_not_a_fixed_sample_budget(monkeypatch):
    monkeypatch.setattr(shared_source, "TrainingControllerState", _Controller)
    args = SimpleNamespace(dp_size=1, num_train_steps=1, global_batch_size=1, dflash_packing=True)
    hub = shared_source.SharedSourceState(args, {"packed": args})
    hub.load_dataset(args)
    hub.invoke("packed", "submit_training_dataset", (), {})
    assert hub.remaining["packed"] is None
    assert len(hub.get_prompts(6)) == 6
    hub.retire("packed", owners_fenced=True)
    assert hub.remaining["packed"] == 0
    assert hub.get_prompts(1) == []
