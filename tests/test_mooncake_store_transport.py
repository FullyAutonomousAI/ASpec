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

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from aspec.config.mooncake_config import MooncakeConfig
from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore


def test_selected_host_staging_is_not_treated_as_fallback():
    store = EagleMooncakeStore(
        MooncakeConfig(
            data_plane="store",
            protocol="tcp",
            enable_gpu_direct=False,
            allow_host_fallback=False,
        )
    )
    store._initialized = True
    store._init_event.set()
    store._gpu_direct_available = False
    store._gpu_receive_buffer = None
    store.wait_for_keys = MagicMock()
    store._get_tensors_via_host_buffer = MagicMock(
        return_value={"hidden_states": "hs", "input_ids": "ids"}
    )

    target_module = ModuleType("aspec.models.target.eagle3_target_model")

    class Eagle3TargetOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    target_module.Eagle3TargetOutput = Eagle3TargetOutput
    with patch.dict(
        "sys.modules",
        {"aspec.models.target.eagle3_target_model": target_module},
    ):
        output = store.get(
            "sample",
            shapes={"hidden_states": (2, 3), "input_ids": (2,)},
            dtypes={},
            device="cuda",
        )

    assert output.hidden_states == "hs"
    store._get_tensors_via_host_buffer.assert_called_once()


def test_unavailable_gpu_direct_does_not_silently_select_host_staging():
    store = EagleMooncakeStore(MooncakeConfig(enable_gpu_direct=True, allow_host_fallback=False))
    store._initialized = True
    store._init_event.set()
    store._gpu_direct_available = False
    store.wait_for_keys = MagicMock()
    store._get_tensors_via_host_buffer = MagicMock()
    target_module = ModuleType("aspec.models.target.eagle3_target_model")
    target_module.Eagle3TargetOutput = MagicMock()

    with (
        patch.dict("sys.modules", {"aspec.models.target.eagle3_target_model": target_module}),
        pytest.raises(RuntimeError, match="unavailable.*fallback.*disabled"),
    ):
        store.get(
            "sample",
            shapes={"hidden_states": (2, 3), "input_ids": (2,)},
            dtypes={},
            device="cuda",
        )

    store._get_tensors_via_host_buffer.assert_not_called()


@pytest.mark.parametrize("max_seq_length", [4, 8])
def test_store_fa4_cp1_publishes_the_unsharded_consumer_key(max_seq_length):
    import torch

    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    store._initialized = True
    store._init_event.set()
    store._put_raw_tensors = MagicMock()
    hidden = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    ids = torch.arange(8)
    last = hidden[:, :2].contiguous()

    metadata = store.put_direct_cp_shards(
        "sample", hidden, ids, last, cp_size=1, max_seq_length=max_seq_length
    )

    keys, tensors = store._put_raw_tensors.call_args.args
    assert keys == ["sample_hs", "sample_ids", "sample_lhs"]
    assert metadata["mooncake_key"] == "sample"
    assert metadata["shapes"]["hidden_states"] == (max_seq_length, 4)
    assert torch.equal(tensors[0], hidden[:max_seq_length])
    assert torch.equal(tensors[1], ids[:max_seq_length])


@pytest.mark.parametrize("max_seq_length", [4, 8])
def test_transfer_engine_cp1_publishes_a_plain_descriptor(max_seq_length):
    import torch

    store = EagleMooncakeStore(
        MooncakeConfig(
            data_plane="transfer_engine", enable_gpu_direct=True, allow_host_fallback=False
        )
    )
    store._initialized = True
    store._init_event.set()
    store._p2p_transfer = MagicMock()
    from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor, MooncakeP2PTensorDescriptor

    descriptor = MooncakeP2PDescriptor(
        version=3,
        run_id="run",
        attempt_id="attempt",
        store_key="sample",
        data_id="sample",
        request_id="request",
        source_engine="0",
        source_endpoint="source:15000",
        source_release_endpoint="native_notify",
        slot_base_ptr=1024,
        slot_id=0,
        generation=1,
        payload_bytes=2,
        tensors=(MooncakeP2PTensorDescriptor("hidden_states", (1, 1), "bfloat16", 0, 2),),
    )
    store._p2p_transfer.stage.return_value.to_key.return_value = descriptor.to_key()
    hidden = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    ids = torch.arange(8)
    last = hidden[:, :2].contiguous()

    metadata = store.put_direct_cp_shards(
        "sample",
        hidden,
        ids,
        last,
        cp_size=1,
        max_seq_length=max_seq_length,
        data_id="sample",
        request_id="request",
        source_engine="0",
    )

    assert MooncakeP2PDescriptor.from_key(metadata["mooncake_key"]) == descriptor
    stage = store._p2p_transfer.stage.call_args.kwargs
    assert stage["store_key"] == "sample"
    tensors = dict(stage["tensors"])
    assert torch.equal(tensors["hidden_states"], hidden[:max_seq_length])
    assert torch.equal(tensors["input_ids"], ids[:max_seq_length])
    assert torch.equal(tensors["last_hidden_states"], last[:max_seq_length])
    assert stage["data_id"] == "sample"
    assert stage["request_id"] == "request"


def test_store_close_keeps_registered_buffers_alive_until_client_is_closed():
    events = []
    raw = MagicMock()
    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    store._store = raw
    store._initialized = True
    store._init_event.set()
    store._registered_buffers = {100: 64, 200: 64}
    host = MagicMock()
    gpu = MagicMock()
    manager = MagicMock()
    store._host_buffer_pool = host
    store._gpu_send_buffer = gpu
    store._async_put_manager = manager
    manager.shutdown.side_effect = lambda: events.append("drain")
    raw.close.side_effect = lambda: events.append("close")
    gpu.free.side_effect = lambda: events.append("free_gpu")
    host.shutdown.side_effect = lambda: events.append("free_host")

    store.close()
    store.close()

    assert events == ["drain", "close", "free_gpu", "free_host"]
    assert store._store is None
    assert store._registered_buffers == {}
    assert not store._init_event.is_set()


def test_store_close_failure_preserves_registered_allocations_for_retry():
    raw = MagicMock()
    raw.close.side_effect = RuntimeError("client close failed")
    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    store._store = raw
    store._registered_buffers = {100: 64}
    host = MagicMock()
    gpu = MagicMock()
    store._host_buffer_pool = host
    store._gpu_receive_buffer = gpu

    with pytest.raises(RuntimeError, match="client close failed"):
        store.close()

    host.shutdown.assert_not_called()
    gpu.free.assert_not_called()
    assert store._registered_buffers == {100: 64}
    assert store._store is raw

    raw.close.side_effect = None
    store.close()
    host.shutdown.assert_called_once()
    gpu.free.assert_called_once()


def test_failed_host_buffer_registration_cannot_publish_a_ready_store(monkeypatch):
    import torch

    from aspec.transfer.mooncake import store as store_module

    raw = MagicMock()
    raw.setup.return_value = 0
    raw.register_buffer.return_value = -1
    pool = MagicMock()
    pool._buffers = [MagicMock(ptr=100, size=64)]
    store = EagleMooncakeStore(
        MooncakeConfig(data_plane="store", protocol="tcp", async_put_pool_size=1)
    )
    monkeypatch.setattr(store_module, "MooncakeDistributedStore", lambda: raw)
    monkeypatch.setattr(store_module, "HostBufferPool", lambda **_: pool)
    monkeypatch.setattr(store, "_verify_force_delete", lambda: None)
    monkeypatch.setattr(store, "_build_replicate_config", lambda: None)

    with pytest.raises(RuntimeError, match="register.*host buffer"):
        store.setup(device=torch.device("cpu"))

    assert not store._initialized
    assert not store._init_event.is_set()
    raw.close.assert_called_once()
    pool.shutdown.assert_called_once()


@pytest.mark.parametrize("name,dtype", [("hidden_states", "bfloat16"), ("input_ids", "int64")])
def test_host_reads_own_their_bytes_after_transport_buffer_reuse(name, dtype):
    import torch

    source = torch.arange(8, dtype=getattr(torch, dtype))
    expected = source.clone()
    buffer = MagicMock()
    buffer.ptr.return_value = source.data_ptr()
    buffer.size.return_value = source.numel() * source.element_size()
    raw = MagicMock()
    raw.batch_get_buffer.return_value = [buffer]
    store = EagleMooncakeStore(MooncakeConfig(data_plane="store", protocol="tcp"))
    store._store = raw

    result = store._get_tensors_via_host_buffer(
        ["sample"], [(name, tuple(source.shape), source.dtype)], torch.device("cpu")
    )
    source.zero_()  # The transport is free to recycle the returned buffer now.
    torch.testing.assert_close(result[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("unregister_result", [0, -1, RuntimeError("unregister failed")])
def test_gpu_setup_failure_keeps_registered_memory_alive(monkeypatch, unregister_result):
    from aspec.transfer.mooncake import store as store_module

    events = []
    raw = MagicMock()
    raw.register_buffer.return_value = 0
    def unregister(ptr):
        events.append("unregister")
        if isinstance(unregister_result, Exception):
            raise unregister_result
        return unregister_result
    raw.unregister_buffer.side_effect = unregister
    receive = MagicMock(ptr=100, size=64)
    receive.free.side_effect = lambda: events.append("free")
    send = MagicMock(ptr=0, size=0)
    send.initialize.side_effect = RuntimeError("allocation failed")
    monkeypatch.setattr(store_module, "GPUReceiveBuffer", lambda **_: receive)
    monkeypatch.setattr(store_module, "GPUSendBuffer", lambda **_: send)
    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    store._store = raw

    if unregister_result == 0:
        store._setup_gpu_direct()
        assert events == ["unregister", "free"]
        assert store._registered_buffers == {}
        assert store._gpu_receive_buffer is None
        assert store._gpu_send_buffer is None
    else:
        with pytest.raises(RuntimeError, match="unregister.*GPU"):
            store._setup_gpu_direct()
        receive.free.assert_not_called()
        assert store._registered_buffers == {100: 64}
        assert store._gpu_receive_buffer is receive
        store.close()
        receive.free.assert_called_once()
    assert not store._gpu_direct_available


@pytest.mark.parametrize("close_fails", [False, True])
def test_setup_retry_cannot_replace_a_client_owning_registered_memory(monkeypatch, close_fails):
    from aspec.transfer.mooncake import store as store_module

    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    old = MagicMock()
    store._store = old
    receive = MagicMock()
    store._gpu_receive_buffer = receive
    store._registered_buffers = {100: 64}
    if close_fails:
        old.close.side_effect = RuntimeError("close failed")
    def new_client():
        old.close.assert_called_once()
        receive.free.assert_called_once()
        assert not store._registered_buffers
        raise RuntimeError("new client reached")
    factory = MagicMock(side_effect=new_client)
    monkeypatch.setattr(store_module, "MooncakeDistributedStore", factory)
    with pytest.raises(RuntimeError, match="close failed" if close_fails else "new client reached"):
        store.setup()
    if close_fails:
        factory.assert_not_called()
        receive.free.assert_not_called()
        assert store._store is old
        old.close.side_effect = None
        store.close()
