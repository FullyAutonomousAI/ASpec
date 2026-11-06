# Copyright (C) 2026 Tencent. All rights reserved.
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

# Adapted from the retained AngelSpec source snapshot 0966eb139d1c4054767e4a37bfc8a33db5989a72
# tests/test_target_prefill.py. ASpec adjusts namespace bindings and regression coverage.

# Adapted from AngelSpec tests/test_target_prefill.py (Apache-2.0).

import pytest
import torch

from aspec.inference.target_prefill import (
    TargetPrefillDirectSink,
    TargetPrefillRequest,
)
from aspec.transfer.mooncake.p2p import (
    MooncakeP2PDescriptor,
    MooncakeP2PTensorDescriptor,
    direct_cp_p2p_key_for_rank,
)


class _Descriptor:
    def __init__(self, key, payload_bytes):
        self.key = key
        self.payload_bytes = payload_bytes

    def to_key(self):
        return self.key


class _Reservation:
    def __init__(self, key, tensors):
        self.descriptor = _Descriptor(
            key,
            sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        )
        self.tensors = tensors
        self.state = "reserved"

    def commit(self):
        if self.state != "reserved":
            raise RuntimeError("bad commit")
        self.state = "committed"
        return self.descriptor

    def publish(self):
        if self.state != "committed":
            raise RuntimeError("bad publish")
        self.state = "published"

    def discard(self):
        if self.state != "published":
            self.state = "discarded"


class _Store:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.reservations = []

    def reserve_p2p_output(self, **kwargs):
        if self.fail_after is not None and len(self.reservations) == self.fail_after:
            raise RuntimeError("reserve failed")
        token_length = kwargs["sequence_length"]
        feature_length = kwargs.get("feature_sequence_length") or token_length
        hidden_size = kwargs["hidden_size"]
        num_layers = kwargs["num_aux_layers"]
        tensors = {
            "hidden_states": torch.empty(feature_length, hidden_size * num_layers),
            "input_ids": torch.empty(token_length, dtype=torch.int64),
        }
        if kwargs["store_last_hidden_states"]:
            tensors["last_hidden_states"] = torch.empty(feature_length, hidden_size)
        reservation = _Reservation(kwargs["key"], tensors)
        self.reservations.append(reservation)
        return reservation


def _sink(store, *, context_parallel_size=1):
    return TargetPrefillDirectSink(
        store=store,
        requests=(
            TargetPrefillRequest("request-a", "data-a", 2),
            TargetPrefillRequest("request-b", "data-b", 3),
        ),
        capture_layers=(1, 4),
        hidden_size=2,
        store_last_hidden_states=True,
        source_engine="target-0",
        context_parallel_size=context_parallel_size,
    )


def test_direct_sink_writes_variable_length_batch_and_publishes():
    store = _Store()
    sink = _sink(store)
    input_ids = torch.arange(5)
    sink.begin(
        input_ids=input_ids,
        request_ids=("request-a", "request-b"),
        sequence_lengths=(2, 3),
        prefix_lengths=(0, 0),
    )

    hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    residual = torch.full_like(hidden, 100)
    sink.capture(1, hidden, None)
    sink.capture(4, hidden, residual)
    sink.finish(hidden + 200)

    assert sink.commit_request("request-a").startswith("data-a_")
    assert sink.commit_request("request-b").startswith("data-b_")
    assert sink.payload_bytes("request-a") == 64
    sink.publish()

    first, second = store.reservations
    assert torch.equal(first.tensors["input_ids"], input_ids[:2])
    assert torch.equal(second.tensors["input_ids"], input_ids[2:])
    assert torch.equal(first.tensors["hidden_states"][:, :2], hidden[:2])
    assert torch.equal(second.tensors["hidden_states"][:, 2:], hidden[2:] + residual[2:])
    assert torch.equal(first.tensors["last_hidden_states"], hidden[:2] + 200)
    assert all(reservation.state == "published" for reservation in store.reservations)


def test_partial_cp_reservation_failure_releases_already_acquired_slot():
    store = _Store(fail_after=1)
    sink = _sink(store, context_parallel_size=2)
    with pytest.raises(RuntimeError, match="reserve failed"):
        sink.begin(input_ids=torch.arange(5), request_ids=("request-a", "request-b"),
                   sequence_lengths=(2, 3), prefix_lengths=(0, 0))
    assert len(store.reservations) == 1
    assert store.reservations[0].state == "discarded"


def test_partial_response_handoff_keeps_emitted_and_discards_canceled_work():
    store = _Store()
    sink = _sink(store)
    sink.begin(input_ids=torch.arange(5), request_ids=("request-a", "request-b"),
               sequence_lengths=(2, 3), prefix_lengths=(0, 0))
    for layer in (1, 4):
        sink.capture(layer, torch.zeros(5, 2), None)
    sink.finish(torch.ones(5, 2))
    for rid in ("request-a", "request-b"):
        sink.commit_request(rid)
    sink.finalize_request("request-a", emitted=True)
    sink.abort()
    assert [item.state for item in store.reservations] == ["published", "discarded"]


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_direct_sink_writes_cp_shards_and_returns_ranked_bundle(cp_size):
    class DescriptorStore(_Store):
        def reserve_p2p_output(self, **kwargs):
            reservation = super().reserve_p2p_output(**kwargs)
            payload_bytes = reservation.descriptor.payload_bytes
            reservation.descriptor = MooncakeP2PDescriptor(
                version=3,
                run_id="run",
                attempt_id="attempt",
                store_key=kwargs["key"],
                data_id=kwargs["data_id"],
                request_id=kwargs["request_id"],
                source_engine=kwargs["source_engine"],
                source_endpoint="127.0.0.1:1",
                source_release_endpoint="127.0.0.1:2",
                slot_base_ptr=0x100000 + len(self.reservations) * 0x1000,
                slot_id=len(self.reservations) - 1,
                generation=1,
                payload_bytes=payload_bytes,
                tensors=(
                    MooncakeP2PTensorDescriptor(
                        "hidden_states",
                        tuple(reservation.tensors["hidden_states"].shape),
                        "float32",
                        0,
                        reservation.tensors["hidden_states"].numel() * 4,
                    ),
                ),
            )
            return reservation

    store = DescriptorStore()
    sink = TargetPrefillDirectSink(
        store=store,
        requests=(TargetPrefillRequest("request-a", "data-a", 8),),
        capture_layers=(1, 4),
        hidden_size=2,
        store_last_hidden_states=True,
        source_engine="target-0",
        context_parallel_size=cp_size,
    )
    input_ids = torch.arange(8)
    sink.begin(
        input_ids=input_ids,
        request_ids=("request-a",),
        sequence_lengths=(8,),
        prefix_lengths=(0,),
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    sink.capture(1, hidden, None)
    sink.capture(4, hidden, None)
    sink.finish(hidden + 100)

    bundle = sink.commit_request("request-a")
    for rank, reservation in enumerate(store.reservations):
        key = direct_cp_p2p_key_for_rank(bundle, cp_rank=rank, cp_size=cp_size)
        assert MooncakeP2PDescriptor.from_key(key).store_key.endswith(f"_cp{rank}")
        assert torch.equal(reservation.tensors["input_ids"], input_ids)
        assert torch.equal(
            reservation.tensors["hidden_states"][:, :2],
            hidden[len(input_ids) * rank // cp_size : len(input_ids) * (rank + 1) // cp_size],
        )
    assert torch.equal(
        store.reservations[-1].tensors["last_hidden_states"],
        hidden[len(input_ids) * (cp_size - 1) // cp_size :] + 100,
    )
    sink.publish()
    assert all(item.state == "published" for item in store.reservations)


def test_direct_sink_rejects_prefix_and_releases_partial_reservations():
    sink = _sink(_Store())
    with pytest.raises(ValueError, match="prefix-free"):
        sink.begin(
            input_ids=torch.arange(5),
            request_ids=("request-a", "request-b"),
            sequence_lengths=(2, 3),
            prefix_lengths=(0, 1),
        )

    store = _Store(fail_after=1)
    sink = _sink(store)
    with pytest.raises(RuntimeError, match="reserve failed"):
        sink.begin(
            input_ids=torch.arange(5),
            request_ids=("request-a", "request-b"),
            sequence_lengths=(2, 3),
            prefix_lengths=(0, 0),
        )
    assert store.reservations[0].state == "discarded"


def test_direct_sink_publish_is_atomic_and_abort_releases_every_slot():
    store = _Store()
    sink = _sink(store)
    sink.begin(
        input_ids=torch.arange(5),
        request_ids=("request-a", "request-b"),
        sequence_lengths=(2, 3),
        prefix_lengths=(0, 0),
    )
    hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    sink.capture(1, hidden, None)
    sink.capture(4, hidden, None)
    sink.finish(hidden)

    sink.commit_request("request-a")
    with pytest.raises(RuntimeError, match="before every request committed"):
        sink.publish()
    assert all(reservation.state != "published" for reservation in store.reservations)

    sink.abort()
    sink.abort()
    assert all(reservation.state == "discarded" for reservation in store.reservations)


def test_native_mtp_sink_publishes_final_postnorm_once_without_teacher_duplicate():
    store = _Store()
    sink = TargetPrefillDirectSink(
        store=store, requests=(TargetPrefillRequest("request", "data", 5),),
        capture_layers=(3,), hidden_size=2, store_last_hidden_states=False,
        context_parallel_size=1, final_hidden_as_aux=True,
    )
    ids = torch.arange(5)
    sink.begin(input_ids=ids, request_ids=("request",), sequence_lengths=(5,), prefix_lengths=(0,))
    after_norm = torch.arange(10).reshape(5, 2).float()
    # No intermediate auxiliary capture is needed or requested for native MTP.
    sink.finish(after_norm)
    sink.commit_request("request")
    sink.publish()
    tensors = store.reservations[0].tensors
    assert set(tensors) == {"hidden_states", "input_ids"}
    torch.testing.assert_close(tensors["hidden_states"], after_norm, atol=0, rtol=0)
