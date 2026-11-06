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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aspec.transfer.mooncake.p2p import (
    GenerationSlotRing,
    MooncakeP2PDescriptor,
    MooncakeP2PTensorDescriptor,
    MooncakeP2PTransfer,
    direct_cp_p2p_key_for_rank,
    pack_direct_cp_p2p_keys,
    shared_source_cp_views,
)


def descriptor(length=7):
    return MooncakeP2PDescriptor(
        version=3, run_id="run", attempt_id="attempt", store_key="physical-source",
        data_id="0", request_id="0", source_engine="target",
        source_endpoint="node:1", source_release_endpoint="node:2",
        slot_base_ptr=65536, slot_id=0, generation=1, payload_bytes=512 + length * 8,
        tensors=(
            MooncakeP2PTensorDescriptor("hidden_states", (length, 4), "bfloat16", 0, length * 8),
            MooncakeP2PTensorDescriptor("input_ids", (length,), "int64", 256, length * 8),
            MooncakeP2PTensorDescriptor("last_hidden_states", (length, 4), "bfloat16", 512, length * 8),
        ),
    )


@pytest.mark.parametrize("size", [2, 3, 4, 8])
def test_shared_cp_views_keep_one_source_owner_and_transfer_only_owned_rows(size):
    original = descriptor()
    views = shared_source_cp_views(original, "draft", size)
    key = pack_direct_cp_p2p_keys(view.to_key() for view in views)
    total_rows = 0
    for rank, view in enumerate(views):
        assert view.store_key == original.store_key
        assert (view.slot_id, view.generation, view.slot_base_ptr) == (0, 1, 65536)
        assert MooncakeP2PDescriptor.from_key(
            direct_cp_p2p_key_for_rank(key, cp_rank=rank, cp_size=size)
        ) == view
        lo, hi = 7 * rank // size, 7 * (rank + 1) // size
        hs, ids, teacher = view.tensors
        assert hs.shape == teacher.shape == (hi - lo, 4)
        assert ids.shape == (7,)
        assert hs.source_offset == lo * 8
        assert teacher.source_offset == 512 + lo * 8
        assert hs.length == teacher.length == (hi - lo) * 8
        total_rows += hi - lo
        transport = MooncakeP2PTransfer.__new__(MooncakeP2PTransfer)
        transport.config = SimpleNamespace(protocol="nvlink_intra")
        transport.engine = SimpleNamespace(transfer_sync=Mock(return_value=0))
        rc, count = transport._read_payload(view, destination_ptr=131072, opcode="read", notify=None)
        assert rc == 0
        assert count == sum(t.length > 0 for t in view.tensors)
        copied = sum(call.args[3] for call in transport.engine.transfer_sync.call_args_list)
        assert copied == ids.length + hs.length + teacher.length
    assert total_rows == 7


def test_cp_participants_and_independent_experiments_cannot_release_each_other():
    source = descriptor()
    a = shared_source_cp_views(source, "a", 1)
    b = shared_source_cp_views(source, "b", 4)
    ring = GenerationSlotRing(1)
    lease = ring.acquire(source.store_key)
    ring.register_readers(lease, [v.reader_id for v in a + b])
    assert ring.release_reader(lease, "a") is False
    for view in b[:-1]:
        assert ring.release_reader(lease, view.reader_id) is False
    with pytest.raises(TimeoutError):
        ring.acquire("next", timeout=0)
    assert ring.release_reader(lease, b[-1].reader_id) is True
    next_lease = ring.acquire("next")
    assert ring.release_reader(lease, b[-1].reader_id) is False
    assert ring.in_use == 1
    ring.release(next_lease.slot_id, next_lease.generation, next_lease.owner)


def test_legacy_full_reader_is_unchanged():
    original = descriptor()
    view, = shared_source_cp_views(original, "a", 1)
    assert view.tensors == original.tensors
    assert view.payload_bytes == original.payload_bytes
    assert view.reader_id == "a"


def test_partially_sliced_descriptor_is_rejected_before_transfer():
    from dataclasses import replace

    original = descriptor()
    tensors = (replace(original.tensors[0], source_offset=0), *original.tensors[1:])
    with pytest.raises(ValueError, match="every tensor or none"):
        replace(original, tensors=tensors)


@pytest.mark.parametrize("size", [2, 4, 8])
def test_compact_reader_bytes_are_frozen_and_late_ack_cannot_release_new_generation(monkeypatch, size):
    source = descriptor()
    monkeypatch.setenv("ASPEC_RUN_ID", source.run_id)
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", source.attempt_id)
    views = shared_source_cp_views(source, "draft", size)
    transfer = MooncakeP2PTransfer(SimpleNamespace(
        protocol="tcp", allow_host_fallback=False, gpu_buffer_size=1024,
    ))
    transfer.ring = GenerationSlotRing(1)
    transfer.ring.acquire(source.store_key)
    request = dict(run_id=source.run_id, attempt_id=source.attempt_id,
                   slot_id=source.slot_id, generation=source.generation,
                   store_key=source.store_key, payload_bytes=source.payload_bytes)
    payloads = {view.reader_id: view.payload_bytes for view in views}
    subscription = dict(request, operation="subscribe", readers=list(payloads), reader_payload_bytes=payloads)
    transfer._apply_release_request(subscription)
    transfer._apply_release_request(subscription)
    with pytest.raises(RuntimeError, match="cannot change"):
        transfer._apply_release_request(dict(subscription, reader_payload_bytes={
            **payloads, views[0].reader_id: views[0].payload_bytes + 1,
        }))
    ack = dict(request, reader_id=views[0].reader_id, payload_bytes=views[0].payload_bytes)
    transfer._apply_release_request(ack)
    transfer._apply_release_request(ack)
    for view in views[1:]:
        transfer._apply_release_request(dict(request, operation="withdraw", reader_id=view.reader_id))
    metrics = transfer.status()["metrics"]
    assert metrics["shared_expected_reads"] == size
    assert metrics["shared_expected_payload_bytes"] == sum(payloads.values())
    assert metrics["shared_completed_readers"] == {views[0].reader_id: {
        "reads": 1, "payload_bytes": views[0].payload_bytes,
    }}
    assert metrics["shared_withdrawn_payload_bytes"] == sum(v.payload_bytes for v in views[1:])
    assert transfer._shared_reader_bytes == transfer._shared_payload_bytes == {}
    transfer.ring.acquire("next")
    assert transfer._apply_release_request(ack) is False
    assert transfer.ring.in_use == 1


def test_one_cp_ack_keeps_descriptor_until_all_readers_finish():
    from threading import RLock

    from aspec.controller.shared_source import SharedSourceState

    hub = SharedSourceState.__new__(SharedSourceState)
    hub._lock = RLock()
    record = SimpleNamespace(release_outcome=None)
    child = SimpleNamespace(
        release_inference_output=Mock(return_value=True),
        _lifecycle=SimpleNamespace(_records={"work": record}),
    )
    hub.experiments = {"a": child}
    descriptor = object()
    hub._descriptors = {"a": {"work": descriptor}}
    hub.invoke("a", "release_inference_output", (), {"execution_id": "work"})
    assert hub._descriptors["a"]["work"] is descriptor
    record.release_outcome = "consumed"
    hub.invoke("a", "release_inference_output", (), {"execution_id": "work"})
    assert hub._descriptors["a"] == {}
