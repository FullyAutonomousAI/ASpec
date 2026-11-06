# Copyright (C) 2026 Tencent. All rights reserved.
# Adapted from the retained AngelSpec source snapshot 0966eb139d1c4054767e4a37bfc8a33db5989a72
# tests/test_mooncake_p2p.py. Original portions remain Apache-2.0;
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

"""Ownership and descriptor tests for the opt-in GPU P2P data plane."""

import os
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from aspec.config.mooncake_config import MooncakeConfig
from aspec.transfer.mooncake.p2p import (
    GenerationSlotRing,
    MooncakeP2PDescriptor,
    MooncakeP2PReceiveLease,
    MooncakeP2PTensorDescriptor,
    MooncakeP2PTransfer,
    RingLease,
    direct_cp_p2p_key_for_rank,
    is_p2p_key,
    p2p_descriptors_from_key,
    pack_direct_cp_p2p_keys,
)
from aspec.transfer.mooncake.store import MooncakeHiddenStateStore


def _descriptor(**overrides):
    values = {
        "version": 3,
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "store_key": "data-1_key",
        "data_id": "data-1",
        "request_id": "request-1",
        "source_engine": "3",
        "source_endpoint": "10.0.0.1:15001",
        "source_release_endpoint": "10.0.0.1:16001",
        "slot_base_ptr": 0x100000,
        "slot_id": 1,
        "generation": 7,
        "payload_bytes": 40,
        "tensors": (
            MooncakeP2PTensorDescriptor(
                "hidden_states", (2, 4), "bfloat16", 0, 16
            ),
            MooncakeP2PTensorDescriptor("input_ids", (3,), "int64", 16, 24),
        ),
    }
    values.update(overrides)
    return MooncakeP2PDescriptor(**values)


def test_backend_child_preserves_preflight_validated_endpoint(monkeypatch):
    config = MooncakeConfig(
        local_hostname="192.0.2.32",
        data_plane="transfer_engine",
        protocol="tcp",
        gpu_buffer_size=1024,
        enable_gpu_direct=True,
        allow_host_fallback=False,
    )

    class FakeTransfer:
        def __init__(self, received_config):
            self.config = received_config
            self.endpoint = f"{received_config.local_hostname}:15000"

        def setup(self, device, role):
            assert device == torch.device("cpu")
            assert role == "source"

        def status(self):
            return {"endpoint": self.endpoint}

    monkeypatch.setattr(
        "aspec.transfer.mooncake.p2p.MooncakeP2PTransfer", FakeTransfer
    )
    with (
        mock.patch("ray.is_initialized", return_value=False),
        mock.patch(
            "aspec.ray.ray_actor.RayActor.get_node_ip",
            return_value="198.51.100.126",
        ) as get_node_ip,
    ):
        store = MooncakeHiddenStateStore(config)
        store.setup(torch.device("cpu"), p2p_role="source")

    get_node_ip.assert_not_called()
    assert config.local_hostname == "192.0.2.32"
    assert store.get_p2p_status()["endpoint"] == "192.0.2.32:15000"


def test_descriptor_round_trip_is_serializable_and_stable():
    descriptor = _descriptor()
    key = descriptor.to_key()

    assert is_p2p_key(key)
    assert MooncakeP2PDescriptor.from_key(key) == descriptor
    assert descriptor.to_key() == key


@pytest.mark.parametrize("method", ["stage", "reserve"])
def test_source_capacity_wait_does_not_reuse_fetch_retry_timeout(method):
    source = object.__new__(MooncakeP2PTransfer)
    source.role, source.copy_stream = "source", object()
    source.config = SimpleNamespace(get_retry_max_wait_seconds=0.001)
    source.check_last_error = mock.Mock()
    source.ring = mock.Mock()
    source.ring.acquire.side_effect = RuntimeError("stop before GPU copy")
    with pytest.raises(RuntimeError, match="stop before GPU copy"):
        getattr(source, method)(store_key="sample", tensors=[])
    source.ring.acquire.assert_called_once_with("sample")


@pytest.mark.parametrize(
    "override,match",
    [
        ({"attempt_id": ""}, "requires run_id and attempt_id"),
        ({"generation": 0}, "invalid slot lease"),
        ({"payload_bytes": 8}, "exceed payload_bytes"),
        (
            {
                "tensors": (
                    MooncakeP2PTensorDescriptor("x", (1,), "uint8", 8, 8),
                    MooncakeP2PTensorDescriptor("y", (1,), "uint8", 4, 8),
                )
            },
            "invalid P2P tensor extent",
        ),
    ],
)
def test_descriptor_rejects_invalid_extents(override, match):
    with pytest.raises(ValueError, match=match):
        _descriptor(**override)


def test_descriptor_rejects_non_p2p_and_corrupt_keys():
    with pytest.raises(ValueError, match="not a ASpec"):
        MooncakeP2PDescriptor.from_key("ordinary-store-key")
    with pytest.raises(ValueError, match="invalid ASpec"):
        MooncakeP2PDescriptor.from_key("aspec-p2p-v3:not-base64")


@pytest.mark.parametrize("cp_size", [2, 3, 8])
def test_context_parallel_descriptor_bundle_selects_exact_rank(cp_size):
    keys = [
        _descriptor(store_key=f"sample_cp{rank}", slot_id=rank).to_key()
        for rank in range(cp_size)
    ]
    bundle = pack_direct_cp_p2p_keys(keys)

    assert direct_cp_p2p_key_for_rank(bundle, cp_rank=0, cp_size=cp_size) == keys[0]
    assert direct_cp_p2p_key_for_rank(bundle, cp_rank=1, cp_size=cp_size) == keys[1]
    assert direct_cp_p2p_key_for_rank(
        "store-key", cp_rank=1, cp_size=2
    ) == "store-key_cp1"
    assert p2p_descriptors_from_key(bundle) == tuple(
        MooncakeP2PDescriptor.from_key(key) for key in keys
    )


def test_descriptor_expansion_rejects_non_p2p_keys():
    with pytest.raises(ValueError, match="not a ASpec"):
        p2p_descriptors_from_key("ordinary-store-key")


def test_context_parallel_descriptor_bundle_rejects_group_mismatch():
    key = _descriptor(store_key="sample_cp0", slot_id=0).to_key()
    bundle = pack_direct_cp_p2p_keys([key])
    with pytest.raises(ValueError, match="bundle size"):
        direct_cp_p2p_key_for_rank(bundle, cp_rank=0, cp_size=2)


def test_ring_generation_and_stale_ack_fail_fast():
    ring = GenerationSlotRing(1)
    first = ring.acquire("key-a")
    ring.release(first.slot_id, first.generation, first.owner)
    second = ring.acquire("key-b")

    assert second.generation == first.generation + 1
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        ring.release(first.slot_id, first.generation, first.owner)
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        ring.release(second.slot_id, second.generation, "wrong-key")

    ring.release(second.slot_id, second.generation, second.owner)
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        ring.release(second.slot_id, second.generation, second.owner)


def test_ring_backpressure_waits_for_release():
    ring = GenerationSlotRing(1)
    first = ring.acquire("key-a")
    acquired = []

    def waiter():
        acquired.append(ring.acquire("key-b", timeout=1.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    assert acquired == []

    ring.release(first.slot_id, first.generation, first.owner)
    thread.join(timeout=1.0)
    assert len(acquired) == 1
    ring.release(acquired[0].slot_id, acquired[0].generation, acquired[0].owner)
    assert ring.in_use == 0


def test_acknowledged_release_retry_does_not_release_a_new_generation():
    ring = GenerationSlotRing(1)
    first = ring.acquire("key-a")

    assert ring.release_acknowledged(
        first.slot_id, first.generation, first.owner
    ) is True
    second = ring.acquire("key-b")
    assert ring.release_acknowledged(
        first.slot_id, first.generation, first.owner
    ) is False
    assert ring.in_use == 1

    with pytest.raises(RuntimeError, match="stale or foreign"):
        ring.release_acknowledged(second.slot_id, second.generation, "wrong-key")
    ring.release(second.slot_id, second.generation, second.owner)


def test_destination_lease_releases_exactly_once():
    released = []

    class Transfer:
        def _release_destination_lease(self, lease, descriptor):
            released.append((lease, descriptor))

    descriptor = _descriptor()
    ring_lease = RingLease(slot_id=0, generation=3, owner=descriptor.to_key())
    receive_lease = MooncakeP2PReceiveLease(Transfer(), ring_lease, descriptor)

    receive_lease.release()
    assert released == [(ring_lease, descriptor)]
    with pytest.raises(RuntimeError, match="duplicate"):
        receive_lease.release()


def test_transport_status_reports_strict_data_plane_metrics():
    transfer = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="rdma",
            local_hostname="10.0.0.1",
            device_name="mlx5_bond_0",
            rdma_port=1,
            rdma_gid_index=3,
            allow_host_fallback=False,
            gpu_buffer_size=1024,
        )
    )
    transfer.engine = object()
    transfer.endpoint = "10.0.0.1:15000"
    transfer.role = "destination"
    transfer.ring = GenerationSlotRing(2)
    transfer._started_monotonic = time.monotonic()
    transfer._add_metric("destination_reads", 2)
    transfer._add_metric("destination_payload_bytes", 4096)
    transfer._add_metric("destination_transfer_seconds", 0.5)

    status = transfer.status()

    assert status["state"] == "active"
    assert status["activation"] == "strict_rdma_initialized"
    assert status["strict_transport"] is True
    assert status["node_address"] == "10.0.0.1"
    assert status["configured_rdma_route"] == {
        "device_name": "mlx5_bond_0",
        "port": 1,
        "gid_index": 3,
    }
    assert status["observed_rdma_route"] is None
    assert status["notification_worker_error"] is None
    assert status["ring"] == {"slots": 2, "in_use": 0, "slot_bytes": 1024}
    assert status["metrics"]["destination_payload_bytes_per_second"] == 8192.0


def test_tcp_read_payload_is_chunked_below_native_progress_watchdog(monkeypatch):
    transfer = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
        )
    )
    transfer.engine = mock.MagicMock()
    transfer.engine.transfer_sync.return_value = 0
    descriptor = _descriptor(payload_bytes=40)
    notify = object()
    monkeypatch.setattr(
        "aspec.transfer.mooncake.p2p._TCP_TRANSFER_CHUNK_BYTES", 16
    )

    rc, chunks = transfer._read_payload(
        descriptor,
        destination_ptr=0x200000,
        opcode="read",
        notify=notify,
    )

    assert (rc, chunks) == (0, 3)
    assert transfer.engine.transfer_sync.call_args_list == [
        mock.call(descriptor.source_endpoint, 0x200000, 0x100000, 16, "read", None),
        mock.call(descriptor.source_endpoint, 0x200010, 0x100010, 16, "read", None),
        mock.call(descriptor.source_endpoint, 0x200020, 0x100020, 8, "read", notify),
    ]


def test_non_tcp_read_payload_remains_one_operation(monkeypatch):
    transfer = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="rdma",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
        )
    )
    transfer.engine = mock.MagicMock()
    transfer.engine.transfer_sync.return_value = 0
    descriptor = _descriptor(payload_bytes=40)
    monkeypatch.setattr(
        "aspec.transfer.mooncake.p2p._TCP_TRANSFER_CHUNK_BYTES", 16
    )

    assert transfer._read_payload(
        descriptor,
        destination_ptr=0x200000,
        opcode="read",
        notify=None,
    ) == (0, 1)
    transfer.engine.transfer_sync.assert_called_once_with(
        descriptor.source_endpoint,
        0x200000,
        0x100000,
        40,
        "read",
        None,
    )


def test_discarded_source_payload_is_accounted_for_terminal_reconciliation():
    transfer = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
        )
    )
    descriptor = _descriptor(slot_id=0)
    transfer.role = "source"
    transfer.ring = GenerationSlotRing(1)
    lease = transfer.ring.acquire(descriptor.store_key)
    descriptor = _descriptor(
        slot_id=lease.slot_id,
        generation=lease.generation,
    )

    transfer.discard_staged(descriptor)

    assert transfer.ring.in_use == 0
    assert transfer.status()["metrics"]["discarded_source_descriptors"] == 1
    assert transfer.status()["metrics"]["discarded_source_payload_bytes"] == 40


def test_close_fails_closed_when_registered_gpu_memory_cannot_be_unregistered():
    transfer = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
        )
    )
    buffer = mock.MagicMock()
    buffer.data_ptr.return_value = 0x1000
    transfer.buffers = [buffer]
    transfer._ring_buffer = buffer
    transfer.engine = mock.MagicMock()
    transfer.engine.unregister_memory.return_value = -1

    with pytest.raises(RuntimeError, match="GPU memory unregister failed"):
        transfer.close()

    assert transfer.buffers == [buffer]
    assert transfer.engine is not None


def test_acknowledged_release_server_retries_exact_generation(monkeypatch):
    monkeypatch.setenv("ASPEC_RUN_ID", "run-1")
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", "attempt-1")
    source = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
            get_retry_max_wait_seconds=1.0,
        )
    )
    source.role = "source"
    source.ring = GenerationSlotRing(1)
    lease = source.ring.acquire("data-1_key")
    source._start_release_server("127.0.0.1")
    destination = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
            get_retry_max_wait_seconds=1.0,
        )
    )
    descriptor = _descriptor(
        slot_id=lease.slot_id,
        generation=lease.generation,
        source_release_endpoint=source.release_endpoint,
    )

    try:
        destination._acknowledge_source_release(descriptor)
        destination._acknowledge_source_release(descriptor)
    finally:
        source.close()

    assert source.ring.in_use == 0
    assert source.status()["metrics"]["source_releases"] == 1
    assert source.status()["metrics"]["source_release_duplicates"] == 1
    assert destination.status()["metrics"]["destination_release_acks"] == 2


def test_unacknowledged_release_counts_as_transfer_failure():
    destination = MooncakeP2PTransfer(
        SimpleNamespace(
            protocol="tcp",
            allow_host_fallback=False,
            gpu_buffer_size=1024,
            get_retry_max_wait_seconds=0.0,
        )
    )
    descriptor = _descriptor(source_release_endpoint="127.0.0.1:1")

    with pytest.raises(RuntimeError, match="was not acknowledged"):
        destination._acknowledge_source_release(descriptor)

    metrics = destination.status()["metrics"]
    assert metrics["destination_release_failures"] == 1
    assert metrics["transfer_failures"] == 1


def test_shared_release_accounts_successful_reader_separately_from_withdrawal(monkeypatch):
    monkeypatch.setenv("ASPEC_RUN_ID", "run-1")
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", "attempt-1")
    transfer = MooncakeP2PTransfer(SimpleNamespace(protocol="tcp", allow_host_fallback=False,
                                                gpu_buffer_size=1024))
    transfer.ring = GenerationSlotRing(1)
    lease = transfer.ring.acquire("shared")
    request = {"run_id": "run-1", "attempt_id": "attempt-1", "slot_id": lease.slot_id,
               "generation": lease.generation, "store_key": "shared", "payload_bytes": 40}
    transfer._apply_release_request(dict(request, operation="subscribe", readers=["a", "b"]))
    transfer._apply_release_request(dict(request, reader_id="a"))
    transfer._apply_release_request(dict(request, reader_id="a"))
    transfer._apply_release_request(dict(request, operation="withdraw", reader_id="b"))
    metrics = transfer.status()["metrics"]
    assert metrics["shared_completed_readers"] == {"a": {"reads": 1, "payload_bytes": 40}}
    assert metrics["shared_withdrawn_reads"] == 1
    assert transfer.ring.in_use == 0


@pytest.mark.parametrize("slots", [2, 3, 8])
def test_gpu_ring_registers_one_backing_allocation_not_slot_views(monkeypatch, slots):
    import aspec.transfer.mooncake.p2p as p2p

    config = MooncakeConfig(
        protocol="nvlink_intra", data_plane="transfer_engine",
        require_nvlink=True, enable_gpu_direct=True, allow_host_fallback=False,
        gpu_buffer_size=1024, gpu_ring_slots=slots,
    )
    engine = mock.MagicMock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 20001
    engine.register_memory.return_value = 0
    engine.unregister_memory.return_value = 0
    monkeypatch.setattr(p2p, "_import_mooncake_engine", lambda: SimpleNamespace(
        SUPPORT_CUDA=True, SUPPORT_INTRA_NVLINK=True, TransferEngine=lambda: engine,
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: mock.MagicMock())
    cpu_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: cpu_empty(
        *args, **{key: value for key, value in kwargs.items() if key != "device"}
    ))
    transfer = MooncakeP2PTransfer(config)
    with mock.patch.dict("os.environ", {}, clear=True):
        transfer.setup(torch.device("cuda:0"), role="destination")
    backing = transfer._ring_buffer
    engine.register_memory.assert_called_once_with(backing.data_ptr(), slots * 1024, "cuda:0")
    assert len(transfer.buffers) == slots
    for rank, view in enumerate(transfer.buffers):
        assert view.numel() == 1024
        assert view.data_ptr() == backing.data_ptr() + rank * 1024
    transfer.close()
    transfer.close()
    engine.unregister_memory.assert_called_once_with(backing.data_ptr())
    assert transfer._ring_buffer is None and transfer.buffers == []


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="opt-in real NVLink ring registration",
)
def test_real_nvlink_ring_views_unregister_one_cuda_allocation():
    config = MooncakeConfig(
        local_hostname="127.0.0.1", protocol="nvlink_intra",
        data_plane="transfer_engine", require_nvlink=True,
        enable_gpu_direct=True, allow_host_fallback=False,
        gpu_buffer_size=65536, gpu_ring_slots=3,
    )
    transfer = MooncakeP2PTransfer(config)
    transfer.setup(torch.device("cuda:0"), role="destination")
    assert len(transfer.buffers) == 3
    assert all(view._base is transfer._ring_buffer for view in transfer.buffers)
    transfer.close()
    transfer.close()
    assert transfer.engine is None and transfer.buffers == []


def _staging_transfer(monkeypatch, *, drain_error=False):
    from contextlib import nullcontext

    transfer = MooncakeP2PTransfer(SimpleNamespace())
    transfer.role = "source"
    transfer.device = torch.device("cuda", 0)
    transfer.endpoint = "source:1234"
    transfer.ring = GenerationSlotRing(1)
    events = []
    original_release = transfer.ring.release

    def release(*args):
        events.append("release")
        original_release(*args)

    def drain():
        events.append("drain")
        if drain_error:
            raise RuntimeError("stream could not drain")

    transfer.ring.release = release
    transfer.copy_stream = mock.Mock()
    transfer.copy_stream.synchronize.side_effect = drain
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: drain())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: transfer.copy_stream)
    event = mock.Mock()
    event.synchronize.side_effect = lambda: events.append("completed")
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    buffer = mock.MagicMock()
    buffer.numel.return_value = 1024
    buffer.data_ptr.return_value = 0x100000
    buffer.__getitem__.return_value.view.return_value.reshape.return_value.copy_.side_effect = (
        lambda *_args, **_kwargs: events.append("copy")
    )
    transfer.buffers = [buffer]
    tensor = mock.Mock(is_cuda=True, dtype=torch.float32, shape=(4,))
    tensor.numel.return_value = 4
    tensor.element_size.return_value = 4
    tensor.contiguous.return_value = tensor
    return transfer, tensor, events


@pytest.mark.parametrize("drain_error", [False, True])
def test_partial_stage_drains_writes_before_returning_slot(monkeypatch, drain_error):
    transfer, tensor, events = _staging_transfer(monkeypatch, drain_error=drain_error)
    with pytest.raises((ValueError, RuntimeError)):
        transfer.stage(
            store_key="sample", tensors=[("good", tensor), ("bad", torch.zeros(1))],
            run_id="run", attempt_id="attempt",
        )
    assert events == (["copy", "drain"] if drain_error else ["copy", "drain", "release"])
    assert transfer.ring.in_use == int(drain_error)


def test_failed_stage_descriptor_does_not_leak_its_slot(monkeypatch):
    transfer, tensor, _events = _staging_transfer(monkeypatch)
    with pytest.raises(ValueError, match="requires run_id and attempt_id"):
        transfer.stage(store_key="sample", tensors=[("good", tensor)])
    assert transfer.ring.in_use == 0
    assert transfer._metrics["source_descriptors"] == 0


@pytest.mark.parametrize("drain_error", [False, True])
def test_discarded_reservation_drains_producer_writes(monkeypatch, drain_error):
    from aspec.transfer.mooncake.p2p import MooncakeP2PReservation

    transfer, _tensor, events = _staging_transfer(monkeypatch, drain_error=drain_error)
    lease = transfer.ring.acquire("sample")
    descriptor = _descriptor(
        slot_id=lease.slot_id, generation=lease.generation, store_key=lease.owner,
    )
    reservation = MooncakeP2PReservation(transfer, lease, descriptor, {})
    if drain_error:
        with pytest.raises(RuntimeError, match="stream could not drain"):
            reservation.discard()
        assert reservation.state == "reserved"
    else:
        reservation.discard()
        assert reservation.state == "discarded"
    assert events == (["drain"] if drain_error else ["drain", "release"])
    assert transfer.ring.in_use == int(drain_error)
