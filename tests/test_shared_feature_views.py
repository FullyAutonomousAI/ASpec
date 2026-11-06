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

"""Column/row views share a physical owner, not a copied full feature source."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from aspec.feature_contract import (
    FeatureContractError,
    build_target_feature_contract,
    shared_feature_union,
)
from aspec.transfer.mooncake.p2p import (
    MooncakeP2PDescriptor,
    MooncakeP2PTensorDescriptor,
    MooncakeP2PTransfer,
    shared_source_cp_views,
)


def contract(layers, last=True):
    return build_target_feature_contract(drafter_family="eagle3", target_hidden_size=2,
                                          target_num_hidden_layers=12,
                                          aux_hidden_state_layers=layers,
                                          store_last_hidden_states=last)


def test_union_preserves_each_order_and_optional_teacher_without_aliasing():
    first, second = contract([1, 5], False), contract([9, 1])
    union = shared_feature_union([first, second])
    assert union["aux_hidden_state_layers"] == [1, 5, 9]
    assert first["aux_hidden_state_layers"] == [1, 5]
    assert second["aux_hidden_state_layers"] == [9, 1]
    assert "last_hidden_states" in union["tensors"]
    changed = contract([9, 1])
    changed["tensors"]["hidden_states"]["dtype"] = "float32"
    from aspec.feature_contract import _contract_id
    changed["contract_id"] = _contract_id(changed)
    with pytest.raises(FeatureContractError, match="incompatible"):
        shared_feature_union([first, changed])


@pytest.mark.parametrize("size", [1, 2, 4, 9])
def test_layer_major_regions_copy_only_requested_rows_in_requested_order(size):
    length, width, count = 7, 2, 3
    hidden = torch.arange(length * width * count).reshape(length, count, width).bfloat16()
    ids = torch.arange(length)
    source = torch.zeros(512, dtype=torch.uint8)
    physical = hidden.permute(1, 0, 2).contiguous()
    source[:physical.numel() * 2].copy_(physical.view(torch.uint8).flatten())
    source[256:256 + length * 8].copy_(ids.view(torch.uint8))
    descriptor = MooncakeP2PDescriptor(
        version=3, run_id="run", attempt_id="attempt", store_key="physical-source",
        data_id="0", request_id="0", source_engine="target", source_endpoint="node:1",
        source_release_endpoint="node:2", slot_base_ptr=source.data_ptr(), slot_id=0,
        generation=1, payload_bytes=512,
        tensors=(MooncakeP2PTensorDescriptor("hidden_states", (length, count * width),
                                            "bfloat16", 0, length * count * width * 2,
                                            layer_width=width),
                 MooncakeP2PTensorDescriptor("input_ids", (length,), "int64", 256, length * 8)),
    )
    views = shared_source_cp_views(descriptor, "draft", size, source_layers=[1, 5, 9],
                                   requested_layers=[9, 1], tensor_names={"hidden_states", "input_ids"})
    for rank, view in enumerate(views):
        decoded = MooncakeP2PDescriptor.from_key(view.to_key())
        assert decoded == view
        assert view.backing_payload_bytes == descriptor.payload_bytes
        lo, hi = length * rank // size, length * (rank + 1) // size
        destination = torch.zeros(view.payload_bytes, dtype=torch.uint8)
        copied = []
        def transfer(_endpoint, dst, src, n, *_args):
            ctypes.memmove(dst, src, n)
            copied.append(n)
            return 0
        reader = MooncakeP2PTransfer.__new__(MooncakeP2PTransfer)
        reader.config = SimpleNamespace(protocol="nvlink_intra")
        reader.engine = SimpleNamespace(transfer_sync=transfer)
        assert reader._read_payload(view, destination_ptr=destination.data_ptr(), opcode="read", notify=None)[0] == 0
        assert sum(copied) == (hi - lo) * 2 * width * 2 + length * 8
        tensor = view.tensors[0]
        actual = destination[tensor.offset:tensor.offset + tensor.length].view(torch.bfloat16)
        actual = actual.reshape(2, hi - lo, width).permute(1, 0, 2)
        torch.testing.assert_close(actual, hidden[lo:hi, [2, 0]], atol=0, rtol=0)
        assert view.generation == descriptor.generation
    damaged = replace(views[0].tensors[0], source_regions=((descriptor.payload_bytes, 1),))
    with pytest.raises(ValueError, match="source view"):
        replace(views[0], tensors=(damaged, views[0].tensors[1]))
