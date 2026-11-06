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

"""Exercise the actual FSDP tensor traversal, without GPUs or compilation."""

import torch
from torch.distributed.utils import _apply_to_tensors

from aspec.models.ops.direct_cp import DirectCPMetadata


def metadata(anchor=3):
    return DirectCPMetadata(
        anchor_positions=torch.tensor([[anchor]]),
        block_keep_mask=torch.tensor([[True]]),
        context_position_ids=torch.arange(8)[None],
        block_size=2,
        cp_rank=0,
        cp_size=1,
        group=None,
    )


def test_fsdp_layers_share_one_forward_cache():
    original = metadata()
    # FSDP applies this traversal independently to every layer's kwargs.
    layers = [_apply_to_tensors(lambda tensor: tensor, original) for _ in range(5)]
    sentinel = object()
    layers[0]._fa4_cache["sparse"] = sentinel
    for layer in layers:
        assert layer is not original
        assert layer._fa4_cache is original._fa4_cache
        assert layer._fa4_cache.get("sparse") is sentinel
    assert list(original._fa4_cache.values()) == [sentinel]


def test_equal_shape_new_forward_does_not_reuse_old_anchors():
    old, new = metadata(3), metadata(6)
    old._fa4_cache["sparse"] = object()
    assert new._fa4_cache is not old._fa4_cache
    assert new._fa4_cache.get("sparse") is None
    assert new.anchor_positions.tolist() == [[6]]
    old._fa4_cache.thd_inputs["aux"] = object()
    assert not new._fa4_cache.thd_inputs


def test_fsdp_still_visits_metadata_tensors():
    original = metadata()
    seen = []

    def visit(tensor):
        seen.append(tensor)
        return tensor.clone()

    result = _apply_to_tensors(visit, original)
    assert len(seen) == 3
    assert result.anchor_positions is not original.anchor_positions
    torch.testing.assert_close(result.anchor_positions, original.anchor_positions)
    assert result._fa4_cache is original._fa4_cache


def test_combined_kv_preserves_exact_output_and_gradients():
    from aspec.models.ops.direct_cp import direct_cp_attention, direct_cp_attention_from_kv

    torch.manual_seed(13)
    meta = metadata()
    meta.backend = "math"
    q = torch.randn(1, 4, 2, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 10, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    rq, rk, rv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    actual, lse = direct_cp_attention_from_kv(q, k, v, meta)
    expected, ref_lse = direct_cp_attention(
        rq, rk[:, :, :8], rv[:, :, :8], rk[:, :, 8:], rv[:, :, 8:], meta
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0)
    dout, dlse = torch.randn_like(actual), torch.randn_like(lse)
    ((actual * dout).sum() + (lse * dlse).sum()).backward()
    ((expected * dout).sum() + (ref_lse * dlse).sum()).backward()
    for value, ref in zip((q, k, v), (rq, rk, rv)):
        torch.testing.assert_close(value.grad, ref.grad, rtol=0, atol=0)


def test_combined_kv_wrong_ownership_is_rejected():
    import pytest

    from aspec.models.ops.direct_cp import direct_cp_attention_from_kv

    q = torch.zeros(1, 4, 2, 8)
    k = torch.zeros(1, 2, 9, 8)
    with pytest.raises(ValueError, match="ownership"):
        direct_cp_attention_from_kv(q, k, k, metadata())
