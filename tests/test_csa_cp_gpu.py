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


"""Native CP regressions for unequal and empty compressed-context owners."""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from aspec.models.draft.csa import (
    CSAForwardMetadata,
    CSALightningIndexer,
    CSAOverlappingCompressor,
    _csa_compressed_prefix_attention,
    _csa_remote_predecessor_halo,
    build_csa_compression_layout,
    csa_eligible_groups,
)
from aspec.models.ops.direct_cp import DirectCPMetadata, build_context_positions

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched with torchrun and ASPEC_RUN_GPU_INTEGRATION=1",
)


@pytest.fixture(scope="module")
def cp_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield dist.group.WORLD
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
        dist.destroy_process_group()


def _metadata(length, ratio, group):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank(group)
    positions = torch.arange(length, device=device)[None]
    anchors = torch.tensor([[1, 9, 13, 21]] if length > 4 else [[0, 1, 2, 3]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    rows = build_context_positions(length, rank, 2, alignment=ratio).to(device)
    local = positions.index_select(1, rows)
    full = CSAForwardMetadata(
        anchor_positions=anchors, block_keep_mask=keep, block_size=2, context_window=4
    )
    meta = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=2,
        context_window=4,
        context_physical_position_ids=local,
    )
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=local,
        block_size=2,
        cp_rank=rank,
        cp_size=2,
        group=group,
        sequence_length=length,
    )
    return (
        device,
        rank,
        rows,
        full,
        meta,
        direct,
        build_csa_compression_layout(full, positions, ratio),
        build_csa_compression_layout(meta, local, ratio),
    )


@pytest.mark.parametrize("length", [4, 24, 28])
@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
def test_csa_indexer_matches_unsharded(cp_group, length, stage):
    torch.manual_seed(313)
    device, rank, rows, full, meta, direct, full_layout, layout = _metadata(length, 4, cp_group)
    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        compress_ratios=[4],
        compress_rope_theta=40000.0,
        rms_norm_eps=1e-6,
    )
    indexer = CSALightningIndexer(config, 0).to(device)
    reference = copy.deepcopy(indexer)
    context = torch.randn(1, length, 16, device=device)
    draft = torch.randn(1, 8, 16, device=device)
    query = torch.randn(1, 8, 4, 4, device=device)
    teacher_key = torch.randn(1, length // 4, 2, 4, device=device)
    positions = (full.anchor_positions[..., None] + torch.arange(2, device=device)).flatten(1)
    local_context = context.index_select(1, rows)
    halo = _csa_remote_predecessor_halo(
        context_hidden=local_context,
        metadata=meta,
        layout=layout,
        ratio=4,
        direct_cp_metadata=direct,
    )
    call = (
        indexer.forward_distributed_warmup
        if stage == "indexer_warmup"
        else indexer.forward_distributed_sparse
    )
    loss = call(
        draft_hidden=draft[:, rank * 4 : (rank + 1) * 4],
        context_hidden=local_context,
        draft_position_ids=positions[:, rank * 4 : (rank + 1) * 4],
        layout=layout,
        metadata=meta,
        main_query=query[:, rank * 4 : (rank + 1) * 4],
        main_compressed_key=teacher_key[:, rows.numel() * 0 : 0]
        if rows.numel() == 0
        else teacher_key[:, rows[0].item() // 4 : (rows[-1].item() + 1) // 4],
        scale=0.5,
        workspace_bytes=150,
        direct_cp_metadata=direct,
        remote_predecessor=halo,
    )
    if stage == "sparse":
        _, loss = loss
    eligible = csa_eligible_groups(full, full_layout)
    scores, indices = reference(draft, context, positions, full_layout, eligible)
    expected = reference.loss_per_query(
        index_scores=scores,
        topk_indices=indices,
        main_query=query,
        main_compressed_key=teacher_key,
        eligible=eligible,
        stage=stage,
        scale=0.5,
    )
    loss.sum().backward()
    expected.sum().backward()
    losses = [torch.empty_like(loss) for _ in range(2)]
    dist.all_gather(losses, loss.detach(), group=cp_group)
    grads = []
    for (name, parameter), (_, other) in zip(
        indexer.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=cp_group)
        grads.append((name, parameter.grad, other.grad))
    torch.testing.assert_close(torch.cat(losses, 1), expected, rtol=2e-5, atol=2e-6)
    for name, actual, other in grads:
        torch.testing.assert_close(
            actual, other, rtol=2e-5, atol=8e-6, msg=lambda message: f"{name}: {message}"
        )


@pytest.mark.parametrize("length", [4, 28])
def test_csa_compressor_halo_backward_matches_unsharded(cp_group, length):
    torch.manual_seed(718)
    device, rank, rows, full, meta, direct, full_layout, layout = _metadata(length, 4, cp_group)
    compressor = CSAOverlappingCompressor(
        hidden_size=16, num_heads=2, head_dim=4, ratio=4, rope_theta=40000.0, norm_eps=1e-6
    ).to(device)
    reference = copy.deepcopy(compressor)
    context = torch.randn(1, length, 16, device=device)
    local = context.index_select(1, rows).detach().requires_grad_()
    complete = context.detach().clone().requires_grad_()
    halo = _csa_remote_predecessor_halo(
        context_hidden=local, metadata=meta, layout=layout, ratio=4, direct_cp_metadata=direct
    )
    actual = compressor(local, layout, remote_predecessor=halo)
    expected = reference(complete, full_layout)
    group_rows = rows[::4] // 4
    weights = torch.randn_like(expected[0])
    sum((a * weights.index_select(1, group_rows)).sum() for a in actual).backward()
    sum((a * weights).sum() for a in expected).backward()
    grads = []
    for (name, parameter), (_, other) in zip(
        compressor.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=cp_group)
        grads.append((name, parameter.grad, other.grad))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b.index_select(1, group_rows), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(
        local.grad, complete.grad.index_select(1, rows), rtol=1e-5, atol=2e-6
    )
    for name, a, b in grads:
        torch.testing.assert_close(
            a, b, rtol=2e-5, atol=8e-6, msg=lambda message: f"{name}: {message}"
        )


def test_csa_prefix_empty_owner_matches_full_attention(cp_group):
    torch.manual_seed(191)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank(cp_group)
    positions = torch.arange(16, device=device)[None]
    local_positions = positions[:, :0] if rank == 0 else positions
    anchors = torch.tensor([[9, 13]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    full = CSAForwardMetadata(
        anchor_positions=anchors, block_keep_mask=keep, block_size=1, context_window=4
    )
    meta = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=1,
        context_window=4,
        context_physical_position_ids=local_positions,
    )
    full_layout = build_csa_compression_layout(full, positions, 8)
    layout = build_csa_compression_layout(meta, local_positions, 8)
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=local_positions,
        block_size=1,
        cp_rank=rank,
        cp_size=2,
        group=cp_group,
        sequence_length=16,
    )
    single = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=positions,
        block_size=1,
        cp_rank=0,
        cp_size=1,
        group=None,
        sequence_length=16,
    )
    q = torch.randn(1, 2, 4, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 2, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    local_q = q[:, rank : rank + 1].detach().clone().requires_grad_()
    local_k = (k[:, :0] if rank == 0 else k).detach().clone().requires_grad_()
    local_v = (v[:, :0] if rank == 0 else v).detach().clone().requires_grad_()
    full_q, full_k, full_v = [a.detach().clone().requires_grad_() for a in (q, k, v)]
    actual, lse = _csa_compressed_prefix_attention(
        query_own=local_q,
        compressed_key=local_k,
        compressed_value=local_v,
        metadata=meta,
        layout=layout,
        direct_cp_metadata=direct,
        scale=0.125,
    )
    expected, expected_lse = _csa_compressed_prefix_attention(
        query_own=full_q,
        compressed_key=full_k,
        compressed_value=full_v,
        metadata=full,
        layout=full_layout,
        direct_cp_metadata=single,
        scale=0.125,
    )
    weights = torch.randn_like(q)
    (actual.float() * weights[:, rank : rank + 1]).sum().backward()
    (expected.float() * weights).sum().backward()
    torch.testing.assert_close(actual, expected[:, rank : rank + 1], rtol=0, atol=0)
    torch.testing.assert_close(lse, expected_lse[:, :, rank : rank + 1], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(local_q.grad, full_q.grad[:, rank : rank + 1], rtol=1e-2, atol=2e-3)
    for local, complete in [(local_k, full_k), (local_v, full_v)]:
        torch.testing.assert_close(
            local.grad, complete.grad[:, :0] if rank == 0 else complete.grad, rtol=1e-2, atol=2e-3
        )
