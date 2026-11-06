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

"""Compare QSA CP warmup with the ordinary unsharded teacher and gradients."""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.qsa import QSAForwardMetadata, QSALayer, build_qsa_compression_layout
from aspec.models.ops.direct_cp import DirectCPMetadata, build_context_positions

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched with torchrun and ASPEC_RUN_GPU_INTEGRATION=1",
)


@pytest.fixture(scope="module")
def cp_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield dist.group.WORLD
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
@pytest.mark.parametrize("context_length", [4, 24, 28])
@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_qsa_cp_warmup_matches_unsharded_loss_and_gradients(
    cp_group, ratio, packed, context_length, stage
):
    rank = dist.get_rank(cp_group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(20261031)
    config = DFlashConfig(
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_target_layers=1,
        target_hidden_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        layer_types=["qwen_sparse_attention"],
        qsa_training_stage=stage,
        qsa_token_budget=8,
        qsa_compress_ratio=ratio,
        qsa_index_n_heads=2,
        qsa_index_head_dim=4,
        qsa_rope_dim=2,
        qsa_workspace_bytes=1100,
    )
    layer = QSALayer(config, 0).to(device)
    reference = copy.deepcopy(layer)
    anchors = torch.tensor(
        [[0, 1, 2, 3]] if context_length == 4 else [[1, 9, 13, 21]], device=device
    )
    keep = torch.tensor([[True, True, True, not packed]], device=device)
    positions = torch.arange(context_length, device=device)[None]
    document_length = context_length // 2
    docs = positions // document_length if packed else None
    anchor_docs = anchors // document_length if packed else None
    rope = positions % document_length if packed else positions
    anchor_rope = anchors % document_length if packed else anchors
    draft_positions = (anchor_rope[..., None] + torch.arange(2, device=device)).flatten(1)
    context = torch.randn(1, context_length, 16, device=device)
    draft = torch.randn(1, 8, 16, device=device)
    query = torch.randn(1, 8, 4, 4, device=device)
    key = torch.randn(1, context_length + 8, 2, 4, device=device)
    rows = torch.arange(rank * 4, (rank + 1) * 4, device=device)
    context_rows = build_context_positions(context_length, rank, 2, alignment=ratio).to(device)
    if packed:
        document_rows = build_context_positions(document_length, rank, 2, alignment=ratio).to(
            device
        )
        context_rows = torch.cat((document_rows, document_rows + document_length))
    local_positions = positions.index_select(1, context_rows)
    local_docs = None if docs is None else docs.index_select(1, context_rows)
    local_rope = rope.index_select(1, context_rows)
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=local_positions,
        context_doc_ids=local_docs,
        anchor_doc_ids=anchor_docs,
        block_size=2,
        cp_rank=rank,
        cp_size=2,
        group=cp_group,
        sequence_length=context_length,
    )
    local_meta = QSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=2,
        context_doc_ids=local_docs,
        anchor_doc_ids=anchor_docs,
        context_physical_position_ids=local_positions,
        anchor_rope_position_ids=anchor_rope,
    )
    full_meta = QSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=2,
        context_doc_ids=docs,
        anchor_doc_ids=anchor_docs,
    )
    local_context = context.index_select(1, context_rows).detach().requires_grad_()
    local_draft = draft.index_select(1, rows).detach().requires_grad_()
    full_context = context.detach().clone().requires_grad_()
    full_draft = draft.detach().clone().requires_grad_()
    _, loss, _ = layer(
        draft_hidden=local_draft,
        context_hidden=local_context,
        draft_position_ids=draft_positions.index_select(1, rows),
        metadata=local_meta,
        layout=build_qsa_compression_layout(local_meta, local_rope, ratio),
        main_query=query.index_select(1, rows),
        main_key=torch.cat(
            (
                key[:, :context_length].index_select(1, context_rows),
                key[:, context_length:].index_select(1, rows),
            ),
            dim=1,
        ),
        scale=0.5,
        direct_cp_metadata=direct,
    )
    _, expected_loss, _ = reference(
        draft_hidden=full_draft,
        context_hidden=full_context,
        draft_position_ids=draft_positions,
        metadata=full_meta,
        layout=build_qsa_compression_layout(full_meta, rope, ratio),
        main_query=query,
        main_key=key,
        scale=0.5,
    )
    loss.sum().backward()
    expected_loss.sum().backward()
    losses = [torch.empty_like(loss) for _ in range(2)]
    dist.all_gather(losses, loss.detach(), group=cp_group)
    parameter_grads = []
    for (name, parameter), (_, expected) in zip(
        layer.named_parameters(), reference.named_parameters(), strict=True
    ):
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, group=cp_group)
            parameter_grads.append((name, parameter.grad, expected.grad))
    # All collectives finish before assertions, so a failed comparison does not
    # strand the other rank in an unrelated collective.
    torch.testing.assert_close(torch.cat(losses, 1), expected_loss, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(
        local_context.grad, full_context.grad.index_select(1, context_rows), rtol=1e-5, atol=2e-6
    )
    torch.testing.assert_close(
        local_draft.grad, full_draft.grad.index_select(1, rows), rtol=1e-5, atol=2e-6
    )
    # FP64 validation bounds both reduction orders; CP need not round like
    # the unsharded FP32 GEMMs at near-zero parameter-gradient entries.
    for name, actual, expected in parameter_grads:
        print(
            name,
            "max_abs",
            (actual - expected).abs().max().item(),
            "relative_l2",
            ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item(),
            flush=True,
        )
        torch.testing.assert_close(
            actual, expected, rtol=1e-5, atol=8e-6, msg=lambda message: f"{name}: {message}"
        )

    if not packed and stage == "indexer_warmup":
        import torch.nn.functional as F

        weights = {
            name: p.detach().double().requires_grad_() for name, p in reference.named_parameters()
        }

        def norm(x, name):
            return (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                * weights[name]
            )

        def rotary(x, pos):
            # This fixture has rotary_dim=2: the only inverse frequency is exactly one.
            phase = pos.double().unsqueeze(-1).unsqueeze(-1)
            rotated = torch.stack((-x[..., 1], x[..., 0]), -1)
            return torch.cat((x[..., :2] * phase.cos() + rotated * phase.sin(), x[..., 2:]), -1)

        q64 = F.linear(draft[:, ::2].double(), weights["indexer.q_proj.weight"]).view(1, 4, 2, 4)
        q64 = rotary(norm(q64, "indexer.q_norm.weight"), anchors)
        k64 = (
            F.linear(context.double(), weights["indexer.k_proj.weight"])
            .view(1, context_length // ratio, ratio, 4)
            .mean(2)
        )
        k64 = rotary(
            norm(k64, "indexer.k_norm.weight").unsqueeze(2), positions[:, ::ratio]
        ).squeeze(2)
        scores64 = torch.einsum("bqhd,bkd->bqhk", q64, k64).relu().sum(2) * 0.5
        eligible64 = positions[:, ::ratio].unsqueeze(1) + ratio <= anchors.unsqueeze(-1)
        target64 = torch.zeros(1, 4, context_length + 8, device=device, dtype=torch.float64)
        context_visible = positions.unsqueeze(1) < anchors.unsqueeze(-1)
        draft_visible = (
            torch.arange(8, device=device)[None, None] // 2
            == torch.arange(4, device=device)[None, :, None]
        )
        visible64 = torch.cat((context_visible, draft_visible), -1)
        for head in range(4):
            logits64 = (
                torch.einsum(
                    "bqd,bkd->bqk", query[:, ::2, head].double(), key[:, :, head // 2].double()
                )
                * 0.5
            )
            target64 += logits64.masked_fill(~visible64, -torch.inf).softmax(-1)
        target64 = (
            target64[:, :, :context_length].view(1, 4, context_length // ratio, ratio).amax(-1)
            * eligible64
        )
        target64 /= target64.sum(-1, keepdim=True).clamp_min(1e-12)
        valid64 = eligible64.any(-1)
        log64 = scores64.masked_fill(~eligible64, -torch.inf)
        log64 = torch.where(valid64.unsqueeze(-1), log64, torch.zeros_like(log64)).log_softmax(-1)
        log64 = torch.where(eligible64, log64, torch.zeros_like(log64))
        loss64 = F.kl_div(log64, target64, reduction="none").sum(-1).repeat_interleave(2, 1)
        loss64.sum().backward()
        torch.testing.assert_close(torch.cat(losses, 1).double(), loss64, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(expected_loss.double(), loss64, rtol=1e-5, atol=2e-6)
        for name, actual, expected in parameter_grads:
            for gradient in (actual, expected):
                exact = weights[name].grad
                torch.testing.assert_close(gradient.double(), exact, rtol=1e-5, atol=8e-6)
                relative_l2 = (gradient.double() - exact).norm() / exact.norm().clamp_min(1e-30)
                assert relative_l2 < 2e-6, (name, relative_l2.item())
