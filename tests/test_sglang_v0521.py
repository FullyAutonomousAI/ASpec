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
import sys
from types import SimpleNamespace

import pytest
import torch

from aspec.inference.sglang_v0521 import (
    StreamingLayerAccumulator,
    TrainingEngine,
    TrainingFeaturePublisher,
    training_engine_kwargs,
)


@pytest.mark.parametrize("hidden_size", [None, 5120])
def test_training_hook_uses_transport_hidden_size_default(monkeypatch, hidden_size):
    from aspec.config.mooncake_config import MooncakeConfig

    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", SimpleNamespace(
        hook_manager=SimpleNamespace(SUPPORTS_MODEL_HOOK_SETUP=True),
    ))
    args = SimpleNamespace(
        draft_context_parallel_size=1, max_seq_length=2048,
        store_last_hidden_states=True, target_prefill_memory_optimized=True,
        attention_backend="cudnn_flex_attention",
    )
    if hidden_size is not None:
        args.mooncake_hidden_dim = hidden_size
    result = training_engine_kwargs({"aux_hidden_state_layer_ids": [1, 9]}, args)
    actual = result["forward_hooks"][-1]["config"]["hidden_size"]
    assert actual == (MooncakeConfig().hidden_dim if hidden_size is None else hidden_size)


def _publisher(*, layers=(1, 3), direct=False):
    publisher = TrainingFeaturePublisher({
        "layers": list(layers), "hidden_size": 2, "context_parallel_size": 1,
        "max_sequence_length": 32, "store_last_hidden_states": True,
        "direct_prefill": direct, "attention_backend": "cudnn_flex_attention",
    })

    class Store:
        config = SimpleNamespace(data_plane="store", get_retry_max_wait_seconds=5)

        def __init__(self):
            self.calls = []
            self.removals = []
            self.drained = False

        def put(self, **kwargs):
            self.calls.append(kwargs)
            return {"mooncake_key": f"descriptor-{kwargs.get('data_id', kwargs['key'])}"}

        def discard_unpublished_objects(self, keys):
            self.removals.append(list(keys))

        def flush(self):
            pass

        def wait_for_p2p_source_releases(self, timeout):
            self.drained = True
            return True

        def get_p2p_status(self):
            return {"role": "source", "drained": self.drained}

    publisher.store = Store()
    publisher.capture_layers = tuple(layers)
    return publisher


def _batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False, is_extend=lambda: True),
        is_prefill_only=True, extend_prefix_lens_cpu=[0, 0], extend_seq_lens_cpu=[2, 3],
        rids=["r1", "r2"],
        # Does not depend on whether a custom logit processor was requested.
        sampling_info=SimpleNamespace(custom_params=None),
        request_custom_params=[
            {"aspec_data_id": "w1", "aspec_request_id": "r1"},
            {"aspec_data_id": "w2", "aspec_request_id": "r2"},
        ],
    )


def test_publication_preserves_layer_order_rows_teacher_and_identity(monkeypatch):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    publisher = _publisher(layers=(3, 1))
    ids = torch.arange(5)
    aux = [torch.arange(10).reshape(5, 2), torch.arange(10, 20).reshape(5, 2)]
    teacher = torch.arange(20, 30).reshape(5, 2)
    assert publisher.publish(ids, teacher, _batch(), aux) == ["descriptor-w1", "descriptor-w2"]
    first, second = publisher.store.calls
    assert torch.equal(first["hidden_states"], torch.cat(aux[::-1], -1)[:2])
    assert torch.equal(second["hidden_states"], torch.cat(aux[::-1], -1)[2:])
    assert torch.equal(first["last_hidden_states"], teacher[:2])
    assert torch.equal(second["input_ids"], ids[2:])
    assert (first["data_id"], first["request_id"]) == ("w1", "r1")


@pytest.mark.parametrize("failure", ["prefix", "mixed", "identity", "shape", "decode"])
def test_invalid_batch_is_rejected_before_publication(failure):
    publisher = _publisher()
    batch = _batch()
    aux = [torch.ones(5, 2), torch.ones(5, 2)]
    if failure == "prefix":
        batch.extend_prefix_lens_cpu[0] = 1
    elif failure == "mixed":
        batch.request_custom_params[0] = None
    elif failure == "identity":
        batch.request_custom_params[0]["aspec_request_id"] = "other"
    elif failure == "shape":
        aux = [torch.ones(4, 2), torch.ones(5, 2)]
    else:
        batch.is_prefill_only = False
    with pytest.raises(RuntimeError):
        publisher.publish(torch.arange(5), torch.ones(5, 2), batch, aux)
    assert publisher.store.calls == []


def test_native_adapter_preserves_per_request_identity():
    class Engine:
        async def async_generate(self, **kwargs):
            return kwargs

    result = asyncio.run(TrainingEngine(Engine()).async_generate(
        input_ids=[[1, 2], [3, 4]], spec_training_data_id=["w1", "w2"],
        rid=["r1", "r2"], packed_loss_mask=["mask1", "mask2"],
        sampling_params={"max_new_tokens": 0},
    ))
    assert "spec_training_data_id" not in result
    assert result["sampling_params"][1]["custom_params"] == {
        "aspec_data_id": "w2", "aspec_request_id": "r2",
    }


def test_streamed_native_callback_keeps_capture_coordinates_without_clone():
    sink = SimpleNamespace(capture=lambda *args: calls.append(args))
    calls = []
    accumulator = StreamingLayerAccumulator(sink, (2, 4))
    hidden = torch.ones(5, 2)
    accumulator.capture(hidden, owned=False)
    accumulator.capture(hidden + 1, owned=True)
    assert len(accumulator) == 2
    assert calls[0][0] == 2 and calls[0][1] is hidden and calls[0][2] is None
    assert calls[1][0] == 4
    with pytest.raises(RuntimeError, match="extra captured"):
        accumulator.capture(hidden)


def test_configured_hook_uses_native_metadata_and_avoids_host_hidden_output(monkeypatch):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(
        get_parallel=lambda: SimpleNamespace(attn_tp_rank=0),
    ))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.logits_processor", SimpleNamespace(
        LogitsProcessorOutput=SimpleNamespace,
    ))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.layernorm_sp", SimpleNamespace(
        maybe_exit_gather=lambda **kwargs: (kwargs["hidden_states"], kwargs["hidden_states_before_norm"]),
        runs_sp=lambda mode: False,
        sp_exit_gather=lambda hidden, count: hidden,
    ))

    class Processor(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("Training must not invoke the vocabulary head")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.logits_processor = Processor()

        def set_eagle3_layers_to_capture(self, layers):
            self.model.layers_to_capture = [layer + 1 for layer in layers]

        def forward(self, input_ids, positions, forward_batch):
            return self.logits_processor(
                input_ids, torch.ones(5, 2), None, forward_batch,
                [torch.zeros(5, 2), torch.ones(5, 2)],
            )

    model = Model()
    publisher = _publisher()
    publisher.configure_model(model)
    assert publisher.capture_layers == (2, 4)
    # The real eager runner uses .forward(), not nn.Module.__call__().
    result = model.forward(torch.arange(5), None, _batch())
    assert result.next_token_logits is None
    # Scheduler appends one element, tokenizer extends its per-request list.
    assert result.customized_info == {
        "spec_training_mooncake_store_keys": ["descriptor-w1", "descriptor-w2"],
    }
    request_values = []
    request_values.append(result.customized_info["spec_training_mooncake_store_keys"][0])
    accumulated = []
    accumulated.extend(request_values)
    assert accumulated == ["descriptor-w1"]
    assert publisher.forward_started_ns is None
    assert publisher.drain()
    assert '"drained":true' in publisher.transport_status_message()


def test_always_call_hook_discards_unpublished_sink_and_clears_backbone():
    publisher = _publisher(direct=True)
    aborted = []
    publisher.sink = SimpleNamespace(abort=lambda: aborted.append(True))
    publisher.backbone = SimpleNamespace(target_prefill_direct_sink=publisher.sink)
    assert publisher(None, (), None) is None
    assert aborted == [True]
    assert publisher.sink is None
    assert publisher.backbone.target_prefill_direct_sink is None


def test_buffered_p2p_batch_failure_discards_unpublished_reservations(monkeypatch):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    publisher = _publisher()
    publisher.store.config.data_plane = "transfer_engine"
    reservations = []

    class Reservation:
        state = "reserved"
        tensors = {
            "input_ids": torch.empty(2, dtype=torch.int64),
            "hidden_states": torch.empty(2, 4),
            "last_hidden_states": torch.empty(2, 2),
        }

        def discard(self):
            self.state = "discarded"

    def reserve(**kwargs):
        if reservations:
            raise RuntimeError("second request reservation failed")
        item = Reservation()
        reservations.append(item)
        return item

    publisher.store.reserve_p2p_output = reserve
    with pytest.raises(RuntimeError, match="second request reservation failed"):
        publisher.publish(torch.arange(5), torch.ones(5, 2), _batch(),
                          [torch.zeros(5, 2), torch.ones(5, 2)])
    assert reservations[0].state == "discarded"
    assert publisher.sink is None


def test_pending_output_ownership_waits_for_native_ipc_handoff():
    publisher = _publisher()
    finalized = []
    sink = SimpleNamespace(
        validate_publishable=lambda: None,
        finalize_request=lambda rid, emitted: finalized.append((rid, emitted)),
        abort=lambda: pytest.fail("Completed forward must not discard staged output"),
    )
    publisher.sink = sink
    publisher.stage_outputs(["r1", "r2"], ["key1", "key2"])
    publisher(None, (), None)
    assert finalized == []
    assert publisher.sink is None
    publisher.resolve_outputs(["r1", "r2"], ["key1"])
    assert finalized == [("r1", True), ("r2", False)]
    assert publisher.pending_outputs == {}


def test_failed_response_or_idle_drain_cancels_unsent_slots():
    publisher = _publisher()
    finalized = []
    sink = SimpleNamespace(finalize_request=lambda rid, emitted: finalized.append((rid, emitted)))
    publisher.pending_outputs["r1"] = (sink, "key1")
    assert publisher.drain()
    assert finalized == [("r1", False)]
    assert publisher.pending_outputs == {}


def test_shutdown_awaits_only_owned_tokenizer_tasks():
    async def scenario():
        event = asyncio.Event()
        owned = asyncio.create_task(event.wait())
        unrelated = asyncio.create_task(event.wait())
        closed = []
        engine = SimpleNamespace(tokenizer_manager=SimpleNamespace(asyncio_tasks={owned}),
                                 shutdown=lambda: closed.append(True))
        try:
            wrapper = TrainingEngine(engine)
            await wrapper.async_shutdown()
            await wrapper.async_shutdown()
            assert owned.cancelled()
            assert not unrelated.done()
            assert closed == [True]
        finally:
            unrelated.cancel()
            await asyncio.gather(unrelated, return_exceptions=True)
    asyncio.run(scenario())

def _store_object_keys(call, *, shards=None, store_last=True):
    bases = [call["key"]] if shards is None else [
        f"{call['key']}_{shards[0]}{rank}" for rank in range(shards[1])
    ]
    suffixes = ("hs", "ids", "lhs") if store_last else ("hs", "ids")
    return [f"{base}_{suffix}" for base in bases for suffix in suffixes]


@pytest.mark.parametrize("failure", ["put", "flush"])
def test_store_batch_failure_discards_all_attempted_objects(monkeypatch, failure):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    publisher = _publisher()
    original_put = publisher.store.put

    def put(**kwargs):
        result = original_put(**kwargs)
        if failure == "put" and len(publisher.store.calls) == 2:
            raise RuntimeError("injected put failure after partial write")
        return result

    publisher.store.put = put
    if failure == "flush":
        publisher.store.flush = lambda: (_ for _ in ()).throw(RuntimeError("injected flush failure"))
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        publisher.publish(torch.arange(5), torch.ones(5, 2), _batch(),
                          [torch.ones(5, 2), torch.ones(5, 2)])
    assert publisher.store.removals == [_store_object_keys(call) for call in publisher.store.calls]
    assert publisher.sink is None
    assert publisher.pending_outputs == {}


@pytest.mark.parametrize("layout,shards", [("plain", None), ("cp1", None), ("cp4", ("cp", 4)), ("usp", ("usp", 4))])
@pytest.mark.parametrize("store_last", [False, True])
def test_store_ipc_handoff_only_discards_unemitted_objects(monkeypatch, layout, shards, store_last):
    publisher = _publisher()
    publisher.store_last = store_last
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    if layout.startswith("cp"):
        monkeypatch.setenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", "1")
        publisher.cp_size = 4 if layout == "cp4" else 1
        publisher.store.put_direct_cp_shards = publisher.store.put
    if layout == "usp":
        publisher.attention_backend = "usp"
        for name, value in (("SP_SIZE", "4"), ("RING_SIZE", "2"), ("TTT_LENGTH", "1")):
            monkeypatch.setenv(f"ASPEC_USP_{name}", value)
        publisher.store.put_usp_shards = publisher.store.put
    batch = _batch()
    # CP4 admission requires at least four context tokens per request.
    batch.extend_seq_lens_cpu = [4, 4]
    keys = publisher.publish(torch.arange(8), torch.ones(8, 2), batch,
                             [torch.ones(8, 2), torch.ones(8, 2)])
    publisher(None, (), None)
    assert publisher.store.removals == []
    assert len(publisher.pending_outputs) == 2
    publisher.resolve_outputs(batch.rids, [keys[0]])
    assert publisher.store.removals == [
        _store_object_keys(publisher.store.calls[1], shards=shards, store_last=store_last)
    ]
    assert publisher.pending_outputs == {}
    assert publisher.drain()


def test_store_cleanup_failure_keeps_tracking_without_deleting_emitted_sibling(monkeypatch):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    publisher = _publisher()
    keys = publisher.publish(torch.arange(5), torch.ones(5, 2), _batch(),
                             [torch.ones(5, 2), torch.ones(5, 2)])
    removed = []
    unavailable = True

    def discard(objects):
        removed.append(list(objects))
        if unavailable:
            raise RuntimeError("delete unavailable")

    publisher.store.discard_unpublished_objects = discard
    with pytest.raises(RuntimeError, match="delete unavailable"):
        publisher.resolve_outputs(["r1", "r2"], [keys[1]])
    assert set(publisher.pending_outputs) == {"r1"}
    # The native streamer's failure callback repeats cancellation for the batch.
    with pytest.raises(RuntimeError, match="delete unavailable"):
        publisher.resolve_outputs(["r1", "r2"], [])
    assert all(objects == _store_object_keys(publisher.store.calls[0]) for objects in removed)
    unavailable = False
    assert publisher.drain()
    assert publisher.pending_outputs == {}

def test_partial_store_batch_cleanup_failure_is_retried_at_idle_drain(monkeypatch):
    monkeypatch.delenv("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", raising=False)
    publisher = _publisher()
    original_put = publisher.store.put
    unavailable = True

    def put(**kwargs):
        result = original_put(**kwargs)
        if len(publisher.store.calls) == 2:
            raise RuntimeError("partial batch put failed")
        return result

    def discard(objects):
        if unavailable:
            raise RuntimeError("delete unavailable")
        publisher.store.removals.append(list(objects))

    publisher.store.put = put
    publisher.store.discard_unpublished_objects = discard
    with pytest.raises(RuntimeError, match="delete unavailable"):
        publisher.publish(torch.arange(5), torch.ones(5, 2), _batch(),
                          [torch.ones(5, 2), torch.ones(5, 2)])
    assert publisher.sink is not None
    assert len(publisher.sink.requests) == 2
    with pytest.raises(RuntimeError, match="delete unavailable"):
        publisher.drain()
    unavailable = False
    assert publisher.drain()
    assert publisher.sink is None
    assert publisher.store.removals == [_store_object_keys(call) for call in publisher.store.calls]


def test_native_mtp_uses_postnorm_boundary_without_auxiliary_layer_translation(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(
        get_parallel=lambda: SimpleNamespace(attn_tp_rank=0),
    ))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.logits_processor", SimpleNamespace(
        LogitsProcessorOutput=SimpleNamespace,
    ))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.layernorm_sp", SimpleNamespace(
        maybe_exit_gather=lambda **kw: (kw["hidden_states"], None),
        runs_sp=lambda _mode: False,
        sp_exit_gather=lambda value, _count: value,
    ))
    class Processor(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("MTP feature capture must not project verifier logits")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.logits_processor = Processor()
        def set_dflash_layers_to_capture(self, layers):
            raise AssertionError("Final postnorm is not an intermediate auxiliary layer")
        def forward(self, input_ids, positions, forward_batch):
            return self.logits_processor(input_ids, torch.full((5, 2), 7.0), None, forward_batch)
    publisher = _publisher(layers=(31,))
    publisher.final_hidden_as_aux = True
    publisher.store_last = False
    model = Model()
    publisher.configure_model(model)
    model.forward(torch.arange(5), None, _batch())
    # Native multimodal wrappers may forward (postnorm, direct_sink) unchanged
    # when auxiliary capture is disabled. It is not an extra feature tensor.
    sink = SimpleNamespace()
    publisher.sink = sink
    def publish(_ids, teacher, _batch, aux):
        assert isinstance(teacher, torch.Tensor) and torch.all(teacher == 9)
        assert aux is sink
        publisher.sink = None
        return ["descriptor-w1", "descriptor-w2"]
    publisher.publish = publish
    model.logits_processor(torch.arange(5), (torch.full((5, 2), 9.0), sink), None, _batch())
    assert publisher.capture_layers == (31,)
    for value in publisher.store.calls:
        assert torch.all(value["hidden_states"] == 7)
        assert value["last_hidden_states"] is None
