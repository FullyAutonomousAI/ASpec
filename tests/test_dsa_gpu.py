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

"""Opt-in real FA4 DSA math, detached indexer and 128K workspace checks."""

import os

import pytest
import torch

from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.dsa import (
    DSAForwardMetadata,
    DSALayer,
    build_dsa_token_layout,
    dsa_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 for actual FA4/CUDA checks",
)


def _values():
    device = "cuda"
    length = int(os.getenv("ASPEC_DSA_CONTEXT_LENGTH", "32"))
    if length < 16:
        raise ValueError("DSA GPU context length must be >=16")
    anchors = torch.tensor([[length // 4, length // 2, length * 3 // 4]], device=device)
    docs = torch.arange(length, device=device)[None].div(length // 2, rounding_mode="floor")
    # Two genuine packed documents: the second anchor has an empty context.
    metadata = DSAForwardMetadata(anchors, torch.ones_like(anchors).bool(), 2,
                                  docs, docs.gather(1, anchors))
    context_positions = torch.arange(length, device=device)[None] % (length // 2)
    draft_positions = (context_positions.gather(1, anchors)[..., None]
                       + torch.arange(2, device=device)).flatten(1)
    layout = build_dsa_token_layout(metadata, context_positions, draft_positions,
                                   rope_dim=64, rope_theta=10000.0)
    return length, metadata, layout, draft_positions


def test_selected_token_fa4_matches_independent_attention_and_gradients():
    torch.manual_seed(101)
    length, metadata, layout, _ = _values()
    query = torch.randn(1, 6, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, length + 6, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    selected = torch.tensor([[[0, 1], [2, 3], [-1, -1], [-1, -1],
                              [length // 2, length // 2 + 1],
                              [length // 2 + 2, length // 2 + 3]]], device="cuda")
    actual = dsa_sparse_attention(query=query, key=key, value=value, metadata=metadata,
                                  layout=layout, selected=selected, scale=128**-0.5,
                                  workspace_bytes=1 << 20)
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)]
    rq, rk, rv = reference_inputs
    rows = []
    for row in range(6):
        indices = selected[0, row]
        own = length + row // 2 * 2 + torch.arange(2, device="cuda")
        indices = torch.cat((indices[indices >= 0], own))
        scores = torch.einsum("hd,kd->hk", rq[0, row].float(), rk[0, indices, 0].float()) * 128**-0.5
        rows.append((scores.softmax(-1) @ rv[0, indices, 0].float()).to(query.dtype))
    expected = torch.stack(rows)[None]
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    if relative >= 0.01:
        raise AssertionError(f"DSA FA4 output relative L2 {relative.item()}")
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    for tensor, ref in zip((query, key, value), reference_inputs, strict=True):
        relative = (tensor.grad.float() - ref.grad.float()).norm() / ref.grad.float().norm()
        if relative >= 0.03:
            raise AssertionError(f"DSA FA4 gradient relative L2 {relative.item()}")


def test_empty_compact_partition_keeps_zero_gradient_edges():
    from aspec.models.ops.compact_varlen import compact_varlen_attention

    query = torch.randn(1, 2, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    observed = {}
    output, lse = compact_varlen_attention(
        query=query, key=key, value=value,
        token_indices=torch.zeros(1, 2, 2, device="cuda", dtype=torch.long),
        token_valid=torch.zeros(1, 2, 2, device="cuda", dtype=torch.bool),
        sequence_keep=torch.zeros(1, 2, device="cuda", dtype=torch.bool),
        queries_per_sequence=1, scale=128**-0.5, workspace_bytes=1 << 20,
        return_lse=True, observation=observed,
    )
    if not output.requires_grad or not lse.requires_grad or not observed["empty_local_partition"]:
        raise AssertionError("Empty owner lost its collective backward graph/observation")
    output.sum().backward()
    for tensor in (query, key, value):
        if tensor.grad is None or torch.count_nonzero(tensor.grad):
            raise AssertionError("Empty owner must propagate a real zero gradient")


def test_dense_dsa_layer_observes_real_fixed_attention_backward():
    from aspec.models.draft.dflash import DFlashAttention
    from aspec.models.ops.direct_cp import DirectCPMetadata

    length, metadata, layout, positions = _values()
    config = DFlashConfig(
        hidden_size=256, intermediate_size=512, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, num_hidden_layers=1, layer_types=["deepseek_sparse_attention"],
        dsa_training_stage="dense", dsa_rope_dim=64,
    )
    layer = DFlashAttention(config, 0).cuda().bfloat16()
    direct = DirectCPMetadata(
        anchor_positions=metadata.anchor_positions, block_keep_mask=metadata.block_keep_mask,
        context_position_ids=torch.arange(length, device="cuda")[None],
        context_doc_ids=metadata.context_doc_ids, anchor_doc_ids=metadata.anchor_doc_ids,
        context_rope_position_ids=layout.rope_positions, block_size=2,
        cp_rank=0, cp_size=1, group=None, sequence_length=length, backend="fa4",
    )
    output = layer(
        draft_hidden=torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16),
        context_hidden=torch.randn(1, length, 256, device="cuda", dtype=torch.bfloat16),
        draft_position_ids=positions, context_position_ids=layout.rope_positions,
        direct_cp_metadata=direct, dsa_metadata=metadata,
    )
    if layer.attention_observation["backward_observed"]:
        raise AssertionError("Forward alone cannot certify backward")
    output.float().square().sum().backward()
    if not layer.attention_observation["backward_observed"]:
        raise AssertionError("Merged DSA observation lost a completed fixed backward")


@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
def test_real_bf16_indexer_backward_and_long_context_workspace(stage):
    torch.manual_seed(99)
    length, metadata, layout, positions = _values()
    config = DFlashConfig(
        hidden_size=256, intermediate_size=512, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, num_hidden_layers=1, layer_types=["deepseek_sparse_attention"],
        dsa_training_stage=stage, dsa_token_budget=min(2048, length // 4),
        dsa_index_n_heads=2, dsa_index_head_dim=128, dsa_q_lora_rank=64,
        dsa_rope_dim=64, dsa_workspace_bytes=16 << 20,
    )
    layer = DSALayer(config, 0).cuda().bfloat16()
    draft = torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    context = torch.randn(1, length, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    main_q = torch.randn(1, 6, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    main_k = torch.randn(1, length + 6, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    selected, loss, observed = layer(
        draft_hidden=draft, context_hidden=context, draft_position_ids=positions,
        metadata=metadata, layout=layout, main_query=main_q, main_key=main_k, scale=128**-0.5,
    )
    loss.sum().backward()
    if not torch.isfinite(loss).all():
        raise AssertionError("non-finite DSA indexer KL")
    if any(tensor.grad is not None for tensor in (draft, context, main_q, main_k)):
        raise AssertionError("DSA indexer KL leaked gradients to its inputs/teacher")
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in layer.parameters()):
        raise AssertionError("DSA indexer gradient missing or non-finite")
    if stage == "sparse" and selected.shape != (1, 6, min(2048, length // 4)):
        raise AssertionError("DSA selection was collapsed to block granularity")
    overhead = torch.cuda.max_memory_allocated() - baseline
    if length >= 131072 and overhead > 512 * 1024**2:
        raise AssertionError(f"unbounded DSA indexer workspace: {overhead}")
    print({"context": length, "stage": stage, "peak_extra_bytes": overhead,
           "loss": loss.detach().tolist(), "observation": observed}, flush=True)
