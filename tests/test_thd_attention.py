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

"""Fast CPU regressions; opt-in real FA4/NCCL checks run via torchrun."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from aspec.data.utils import DFlashPackingCollator
from aspec.models.ops import thd_attention as thd
from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    direct_cp_attention,
    gather_owner_sharded_rows,
)


def _documents(lengths, rank=0, size=1):
    result = []
    offset = 0
    for doc, length in enumerate(lengths):
        owned = torch.arange(length * rank // size, length * (rank + 1) // size)
        result.append({
            "input_ids": torch.arange(offset, offset + length)[None],
            "loss_mask": torch.ones(1, length),
            "hidden_states": (owned + offset).float()[None, :, None].expand(-1, -1, 3),
            "last_hidden_states": (owned + offset).float()[None, :, None],
            "_aspec_work": {"work_id": f"w{doc}", "execution_id": f"e{doc}"},
        })
        offset += length
    return result


@pytest.mark.parametrize("size", [1, 2, 4])
def test_unpadded_packing_preserves_per_document_context_ownership(size):
    lengths = [9, 17, 3]
    positions = []
    for rank in range(size):
        batch = DFlashPackingCollator(
            40, pad_to_max_length=False,
            context_parallel_rank=rank, context_parallel_size=size,
        )(_documents(lengths, rank, size))
        assert batch["input_ids"].shape == (1, 29)
        assert batch["base_position_ids"].tolist() == [
            list(range(9)) + list(range(17)) + list(range(3))
        ]
        owned = batch.get("packed_context_positions", torch.arange(29)[None])
        positions.append(owned[0])
        torch.testing.assert_close(batch["hidden_states"][0, :, 0], owned[0].float())
        assert [item["execution_id"] for item in batch["_aspec_work"]] == ["e0", "e1", "e2"]
    torch.testing.assert_close(torch.cat(positions).sort().values, torch.arange(29))


def test_packing_fails_closed_for_wrong_local_features():
    with pytest.raises(ValueError, match="CP ownership"):
        DFlashPackingCollator(
            40, pad_to_max_length=False,
            context_parallel_rank=0, context_parallel_size=2,
        )(_documents([9, 17]))


def _metadata(rank=0, size=1, device="cpu", backend="fa4"):
    lengths = [9, 17, 7]
    context = _documents(lengths, rank, size)
    batch = DFlashPackingCollator(
        40, pad_to_max_length=False,
        context_parallel_rank=rank, context_parallel_size=size,
    )(context)
    positions = batch.get("packed_context_positions", torch.arange(33)[None]).to(device)
    docs = batch["ctx_doc_ids"].to(device)
    anchors = torch.tensor([[0, 4, 7, 9, 12, 20, 28, 31]], device=device)
    keep = torch.tensor([[True, True, False, True, True, True, True, True]], device=device)
    return DirectCPMetadata(
        anchors, keep, positions, 2, rank, size,
        dist.group.WORLD if size > 1 else None,
        context_doc_ids=docs.index_select(1, positions[0]),
        anchor_doc_ids=docs.gather(1, anchors),
        context_rope_position_ids=batch["base_position_ids"].to(device).index_select(1, positions[0]),
        sequence_length=33, backend=backend,
    )


def _mock_fa4(monkeypatch):
    def fixed(q, k, v, **kwargs):
        anchors, keep, blocks, docs, key_docs, positions, key_blocks, key_keep = [
            x[0] for x in kwargs["aux_tensors"]
        ]
        visible = ((positions[None] >= 0) & (positions[None] < anchors[:, None])) | (
            (key_blocks[None] >= 0) & (key_blocks[None] == blocks[:, None])
        )
        visible &= (docs[:, None] == key_docs[None]) & keep[:, None].bool() & key_keep[None].bool()
        repeats = q.shape[2] // k.shape[2]
        kk, vv = k[0].repeat_interleave(repeats, 1), v[0].repeat_interleave(repeats, 1)
        scores = torch.einsum("qhd,khd->hqk", q[0], kk) * kwargs["softmax_scale"]
        scores = scores.masked_fill(~visible[None], -1.0e30)
        lse = scores.logsumexp(-1)
        out = torch.einsum("hqk,khd->qhd", (scores - lse[..., None]).exp(), vv)
        return out[None], lse[None]

    monkeypatch.setitem(
        sys.modules, "flash_attn.cute.interface", SimpleNamespace(flash_attn_func=fixed)
    )
    monkeypatch.setattr(thd, "_build_fixed_mask", lambda *_args, **_kwargs: None)


@pytest.mark.parametrize("lse_only", [False, True])
def test_fixed_thd_matches_independent_attention_and_gradients(monkeypatch, lse_only):
    _mock_fa4(monkeypatch)
    torch.manual_seed(47)
    shapes = [(1, 4, 16, 16), (1, 2, 33, 16), (1, 2, 33, 16),
              (1, 2, 16, 16), (1, 2, 16, 16)]
    inputs = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes]
    references = [x.detach().clone().requires_grad_() for x in inputs]
    metadata = _metadata()
    metadata.fa4_sparse = False
    actual, lse = direct_cp_attention(*inputs, metadata)
    ref_meta = _metadata(backend="math")
    expected, expected_lse = direct_cp_attention(*references, ref_meta)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(lse, expected_lse)
    dout = torch.randn_like(actual) if not lse_only else torch.zeros_like(actual)
    dlse = torch.randn_like(lse)
    ((actual * dout).sum() + (lse * dlse).sum()).backward()
    ((expected * dout).sum() + (expected_lse * dlse).sum()).backward()
    for value, ref in zip(inputs, references):
        torch.testing.assert_close(value.grad, ref.grad)
    assert metadata.attention_observation["qkv_format"] == "thd"
    assert metadata.attention_observation["fa4_mode"] == "fixed"
    assert metadata.attention_observation["backward_observed"] is True


def test_no_active_queries_returns_differentiable_zero(monkeypatch):
    _mock_fa4(monkeypatch)
    metadata = _metadata()
    metadata.fa4_sparse = False
    metadata.block_keep_mask.zero_()
    q = torch.randn(1, 4, 16, 16, requires_grad=True)
    k = torch.randn(1, 2, 49, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out, _ = thd.thd_fa4_attention(q, k, v, metadata, 0.25)
    out.sum().backward()
    assert not out.any() and not q.grad.any() and not k.grad.any() and not v.grad.any()


def test_fsdp_layers_reuse_thd_mask_inputs(monkeypatch):
    from torch.distributed.utils import _apply_to_tensors

    _mock_fa4(monkeypatch)
    meta = _metadata()
    meta.fa4_sparse = False
    original = thd._key_metadata
    calls = []

    def observe(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(thd, "_key_metadata", observe)
    q = torch.randn(1, 4, 16, 16)
    k = torch.randn(1, 2, 49, 16)
    for _ in range(5):
        thd.thd_fa4_attention(q, k, k, _apply_to_tensors(lambda x: x, meta), 0.25)
    assert len(calls) == 1


def _large_metadata(rank, size, device, backend="fa4", layout="contiguous"):
    from aspec.models.ops.direct_cp import build_context_positions

    lengths = [257, 263, 19]
    docs = torch.cat([torch.full((length,), index, device=device) for index, length in enumerate(lengths)])
    positions = torch.cat([
        build_context_positions(length, rank, size, layout=layout) + sum(lengths[:index])
        for index, length in enumerate(lengths)
    ]).to(device)[None]
    anchors = torch.linspace(0, 537, 80, device=device).long()[None]
    keep = torch.ones_like(anchors, dtype=torch.bool)
    keep[:, 13] = False
    return DirectCPMetadata(
        anchors, keep, positions, 7, rank, size, dist.group.WORLD if size > 1 else None,
        context_doc_ids=docs[positions], anchor_doc_ids=docs[anchors],
        sequence_length=539, backend=backend, layout=layout,
    )


def _large_gpu_check(rank, size, device, layout):
    torch.manual_seed(79)
    full = [torch.randn(shape, dtype=torch.bfloat16, device=device) for shape in
            [(1, 4, 560, 128), (1, 2, 539, 128), (1, 2, 539, 128), (1, 2, 560, 128), (1, 2, 560, 128)]]
    meta = _large_metadata(rank, size, device, layout=layout)
    qp = meta.local_block_ids.to(device).repeat_interleave(7) * 7 + torch.arange(7, device=device).repeat(meta.blocks_per_rank)
    selections = [qp, meta.context_position_ids[0], meta.context_position_ids[0], qp, qp]
    inputs = [x.index_select(2, p).detach().requires_grad_() for x, p in zip(full, selections)]
    refs = [x.float().detach().requires_grad_() for x in full]
    from aspec.models.ops.direct_cp import direct_cp_attention_from_kv

    # Native projections/RoPE already own contiguous BTHD KV. Preserve that
    # layout instead of copying a second time into contiguous BHTD storage.
    key = torch.cat((inputs[1], inputs[3]), dim=2).transpose(1, 2).contiguous().transpose(1, 2)
    value = torch.cat((inputs[2], inputs[4]), dim=2).transpose(1, 2).contiguous().transpose(1, 2)
    actual, lse = direct_cp_attention_from_kv(inputs[0], key, value, meta)
    expected, ref_lse = direct_cp_attention(*refs, _large_metadata(0, 1, device, "math"))
    torch.testing.assert_close(actual.float(), expected.index_select(2, qp), atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse, ref_lse.index_select(2, qp), atol=0.025, rtol=0.025)
    dout = torch.randn_like(expected).bfloat16()  # Keep the production dO dtype.
    dlse = torch.randn_like(ref_lse) * 0.1
    ((actual * dout.index_select(2, qp)).sum() + (lse * dlse.index_select(2, qp)).sum()).backward()
    ((expected * dout.float()).sum() + (ref_lse * dlse).sum()).backward()
    for x, ref, indices in zip(inputs, refs, selections):
        torch.testing.assert_close(x.grad.float(), ref.grad.index_select(2, indices), atol=0.04, rtol=0.04)
    assert meta.attention_observation["backward_observed"]


def _distributed_gpu_check():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    torch.manual_seed(57)
    shapes = [(1, 4, 16, 128), (1, 2, 33, 128), (1, 2, 33, 128),
              (1, 2, 16, 128), (1, 2, 16, 128)]
    full = [torch.randn(s, dtype=torch.bfloat16, device=device) for s in shapes]
    meta = _metadata(rank, size, device)
    query_positions = meta.local_block_ids.to(device).repeat_interleave(2) * 2 + torch.arange(2, device=device).repeat(meta.blocks_per_rank)
    ctx_positions = meta.context_position_ids[0]
    selections = [query_positions, ctx_positions, ctx_positions, query_positions, query_positions]
    inputs = [x.index_select(2, indices).detach().requires_grad_() for x, indices in zip(full, selections)]
    reference = [x.float().detach().requires_grad_() for x in full]
    actual, lse = direct_cp_attention(*inputs, meta)
    expected, expected_lse = direct_cp_attention(*reference, _metadata(device=device, backend="math"))
    expected_local = expected.index_select(2, query_positions)
    expected_lse_local = expected_lse.index_select(2, query_positions)
    torch.testing.assert_close(actual.float(), expected_local, atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse.float(), expected_lse_local, atol=0.025, rtol=0.025)
    torch.manual_seed(61)
    dout = torch.randn_like(expected)
    dlse = torch.randn_like(expected_lse) * 0.1
    ((actual.float() * dout.index_select(2, query_positions)).sum()
     + (lse * dlse.index_select(2, query_positions)).sum()).backward()
    ((expected * dout).sum() + (expected_lse * dlse).sum()).backward()
    for x, ref, indices in zip(inputs, reference, selections):
        torch.testing.assert_close(x.grad.float(), ref.grad.index_select(2, indices), atol=0.04, rtol=0.04)

    teacher = torch.arange(33, device=device).float()[None, :, None].expand(-1, -1, 5).contiguous()
    requests = torch.tensor([[rank, 10 + rank, 32 - rank]], device=device)
    gathered = gather_owner_sharded_rows(
        teacher.index_select(1, ctx_positions), ctx_positions[None], requests,
        global_sequence_length=33, group=dist.group.WORLD if size > 1 else None,
    )
    torch.testing.assert_close(gathered, teacher.gather(1, requests[:, :, None].expand(-1, -1, 5)))
    assert meta.attention_observation["backward_observed"] is True
    for layout in ("contiguous", "zigzag"):
        _large_gpu_check(rank, size, device, layout)
    if size > 1:
        dist.barrier()
    if rank == 0:
        print(f"PASS real FA4 fixed THD packed CP{size}: GQA output/LSE/dQ/dK/dV and teacher ownership", flush=True)
    if size > 1:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1", reason="opt-in FA4 GPU check")
def test_real_fa4_thd_cp1():
    _distributed_gpu_check()


if __name__ == "__main__":
    _distributed_gpu_check()


@pytest.mark.parametrize("window,causal", [(None, False), (5, False), (5, True)])
def test_fixed_mask_identity_does_not_specialize_on_packed_context_length(monkeypatch, window, causal):
    _mock_fa4(monkeypatch)
    calls = []
    monkeypatch.setattr(thd, "_build_fixed_mask", lambda *args: calls.append(args))
    for length in (31, 32, 33):
        meta = _metadata()
        meta.fa4_sparse = False
        meta.query_window, meta.draft_is_causal = window, causal
        meta.context_position_ids = meta.context_position_ids[:, :length]
        meta.context_doc_ids = meta.context_doc_ids[:, :length]
        q = torch.randn(1, 4, 16, 16)
        k = torch.randn(1, 2, length + 16, 16)
        thd.thd_fa4_attention(q, k, k, meta, 0.25)
    assert len(set(calls)) == 1, "Packed length must remain runtime metadata, not a FA4 closure constant"
