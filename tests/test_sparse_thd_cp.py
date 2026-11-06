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

"""Opt-in real-FA4/NCCL parity for structured THD context parallelism."""

from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.distributed as dist

from aspec.models.draft.csa import CSAForwardMetadata, build_csa_compression_layout
from aspec.models.draft.dflash import DFlashAttention, DFlashConfig
from aspec.models.draft.dspark import K3DSparkConfig, K3DSparkMLAAttention
from aspec.models.draft.qsa import QSAForwardMetadata, build_qsa_compression_layout
from aspec.models.ops.direct_cp import DirectCPMetadata, build_context_positions


def _context_length() -> int:
    length = int(os.getenv("ASPEC_THD_CONTEXT_LENGTH", "16"))
    dsa = os.getenv("ASPEC_SPARSE_THD_KIND") == "dsa"
    if length < 16 or (length % 4 and not dsa):
        raise ValueError("THD parity context length must be >=16 and divisible by four")
    return length


def _packing_enabled() -> bool:
    return os.getenv("ASPEC_THD_PACKING", "0") == "1"


def _packed_document_length() -> int:
    count = int(os.getenv("ASPEC_THD_PACKED_DOCUMENTS", "4"))
    if count < 1 or _context_length() % count:
        raise ValueError("packed document count must divide the context length")
    return _context_length() // count


def _anchors(device: torch.device) -> torch.Tensor:
    length = _context_length()
    quarter = length // 4
    if _packing_enabled():
        values = [quarter - 2, 2 * quarter - 2, 3 * quarter - 2, length - 2]
    else:
        values = [quarter, 2 * quarter, 3 * quarter, length - 2]
    return torch.tensor([values], device=device)


def _anchor_rope_positions(anchors: torch.Tensor) -> torch.Tensor:
    if not _packing_enabled():
        return anchors
    return anchors.remainder(_packed_document_length())


def _qsa_config(stage: str = "sparse") -> DFlashConfig:
    return DFlashConfig(
        hidden_size=256,
        intermediate_size=512,
        head_dim=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_target_layers=1,
        target_hidden_size=256,
        vocab_size=128,
        max_position_embeddings=max(64, _context_length() + 2),
        layer_types=["qwen_sparse_attention"],
        qsa_training_stage=stage,
        qsa_token_budget=4,
        qsa_compress_ratio=2,
        qsa_index_n_heads=2,
        qsa_index_head_dim=128,
        qsa_rope_dim=64,
        qsa_workspace_bytes=16 * 1024 * 1024,
    )


def _csa_config(stage: str = "sparse") -> DFlashConfig:
    return DFlashConfig(
        hidden_size=256,
        intermediate_size=512,
        head_dim=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_target_layers=1,
        target_hidden_size=256,
        vocab_size=128,
        max_position_embeddings=max(64, _context_length() + 2),
        sliding_window=4,
        layer_types=["compressed_sparse_attention"],
        compress_ratios=[4],
        csa_training_stage=stage,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=128,
        csa_workspace_bytes=16 * 1024 * 1024,
    )


def _mla_config() -> K3DSparkConfig:
    return K3DSparkConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_target_layers=1,
        target_hidden_size=256,
        target_num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=max(64, _context_length() + 2),
        markov_rank=0,
    )


def _metadata(*, rank: int, size: int, device: torch.device):
    length = _context_length()
    anchors = _anchors(device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    positions = build_context_positions(
        length, rank, size, alignment=1 if os.getenv("ASPEC_SPARSE_THD_KIND") == "dsa" else 2
    ).to(device)[None]
    if _packing_enabled():
        document_length = _packed_document_length()
        context_docs = positions.div(document_length, rounding_mode="floor")
        anchor_docs = anchors.div(document_length, rounding_mode="floor")
        rope_positions = positions.remainder(document_length)
    else:
        context_docs = anchor_docs = rope_positions = None
    if os.getenv("ASPEC_THD_EMPTY_OWNER", "0") == "1":
        context_docs = torch.where(positions < length // 2, -1, 1)
        anchor_docs = torch.ones_like(anchors)
        keep = anchors >= length // 2
        rope_positions = positions.remainder(length // 2)
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=positions,
        context_doc_ids=context_docs,
        anchor_doc_ids=anchor_docs,
        context_rope_position_ids=rope_positions,
        block_size=2,
        cp_rank=rank,
        cp_size=size,
        group=dist.group.WORLD if size > 1 else None,
        sequence_length=length,
        backend="fa4",
    )
    sparse = QSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=2,
        context_doc_ids=context_docs,
        anchor_doc_ids=anchor_docs,
        context_physical_position_ids=positions,
        anchor_rope_position_ids=_anchor_rope_positions(anchors),
    )
    layout = build_qsa_compression_layout(
        sparse, rope_positions if rope_positions is not None else positions, ratio=2
    )
    return direct, sparse, layout


def _csa_metadata(*, rank: int, size: int, device: torch.device):
    length = _context_length()
    anchors = _anchors(device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    positions = build_context_positions(
        length, rank, size, alignment=4
    ).to(device)[None]
    if _packing_enabled():
        document_length = _packed_document_length()
        context_docs = positions.div(document_length, rounding_mode="floor")
        anchor_docs = anchors.div(document_length, rounding_mode="floor")
        rope_positions = positions.remainder(document_length)
    else:
        context_docs = anchor_docs = rope_positions = None
    direct = DirectCPMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        context_position_ids=positions,
        context_doc_ids=context_docs,
        anchor_doc_ids=anchor_docs,
        context_rope_position_ids=rope_positions,
        block_size=2,
        cp_rank=rank,
        cp_size=size,
        group=dist.group.WORLD if size > 1 else None,
        sequence_length=length,
        backend="fa4",
        context_window=4,
    )
    sparse = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=keep,
        block_size=2,
        context_window=4,
        context_doc_ids=context_docs,
        anchor_doc_ids=anchor_docs,
        context_physical_position_ids=positions,
    )
    layout = build_csa_compression_layout(
        sparse, rope_positions if rope_positions is not None else positions, ratio=4
    )
    return direct, sparse, layout


def _distributed_qsa_parity(kind="qsa") -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size not in {2, 4}:
        raise RuntimeError("sparse THD parity expects CP2 or CP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20261016)
    stage = os.getenv("ASPEC_THD_STAGE", "sparse")
    config = _qsa_config(stage)
    if kind == "dsa":
        raw = {key: value for key, value in config.to_dict().items() if not key.startswith("qsa_")}
        raw.update(layer_types=["deepseek_sparse_attention"], dsa_training_stage=stage,
                   dsa_token_budget=int(os.getenv("ASPEC_DSA_TOKEN_BUDGET", "4")),
                   dsa_index_n_heads=2, dsa_index_head_dim=128,
                   dsa_q_lora_rank=64, dsa_rope_dim=64,
                   dsa_workspace_bytes=int(os.getenv("ASPEC_DSA_WORKSPACE_BYTES", str(16 << 20))))
        config = DFlashConfig.from_dict(raw)
    attention = DFlashAttention(config).to(
        device=device, dtype=torch.bfloat16
    )
    reference_attention = copy.deepcopy(attention) if rank == 0 else None
    context_full = torch.randn(
        1, _context_length(), 256, device=device, dtype=torch.bfloat16
    )
    draft_full = torch.randn(
        1, 8, 256, device=device, dtype=torch.bfloat16
    )
    direct, sparse, layout = _metadata(rank=rank, size=size, device=device)
    local_blocks = direct.local_block_ids.to(device)
    local_context = context_full.index_select(1, direct.context_position_ids[0]).detach()
    local_context.requires_grad_(True)
    local_draft_rows = (
        local_blocks[:, None] * 2 + torch.arange(2, device=device)[None]
    ).flatten()
    local_draft = draft_full.index_select(1, local_draft_rows).detach()
    local_draft.requires_grad_(True)
    draft_positions = (
        _anchor_rope_positions(direct.anchor_positions)
        .index_select(1, local_blocks)
        .unsqueeze(-1)
        + torch.arange(2, device=device).view(1, 1, -1)
    ).flatten(1)
    if kind == "dsa":
        from aspec.models.draft.dsa import build_dsa_token_layout

        layout = build_dsa_token_layout(
            sparse, direct.context_rope_position_ids if _packing_enabled() else direct.context_position_ids,
            draft_positions, rope_dim=64, rope_theta=10000.0,
        )
    sparse_kwargs = {f"{kind}_metadata": sparse, f"{kind}_layout": layout}
    output = attention(
        draft_hidden=local_draft,
        context_hidden=local_context,
        draft_position_ids=draft_positions,
        context_position_ids=(
            direct.context_rope_position_ids
            if direct.context_rope_position_ids is not None
            else direct.context_position_ids
        ),
        direct_cp_metadata=direct,
        **sparse_kwargs,
    )
    gathered = [torch.empty_like(output) for _ in range(size)]
    dist.all_gather(gathered, output.detach())
    distributed_output = torch.cat(gathered, dim=1)

    local_loss = output.float().square().sum()
    local_loss = local_loss + getattr(attention, f"{kind}_indexer_loss_per_query").float().sum()
    local_loss.backward()
    for parameter in attention.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
    context_lengths = [build_context_positions(
        _context_length(), owner, size, alignment=1 if kind == "dsa" else 2,
    ).numel() for owner in range(size)]
    padded_context_grad = local_context.grad.new_zeros(
        local_context.shape[0], max(context_lengths), local_context.shape[-1],
    )
    padded_context_grad[:, :local_context.shape[1]] = local_context.grad
    context_grad_parts = [torch.empty_like(padded_context_grad) for _ in range(size)]
    draft_grad_parts = [torch.empty_like(local_draft.grad) for _ in range(size)]
    dist.all_gather(context_grad_parts, padded_context_grad)
    dist.all_gather(draft_grad_parts, local_draft.grad)
    distributed_context_grad = torch.cat([
        part[:, :length] for part, length in zip(context_grad_parts, context_lengths, strict=True)
    ], dim=1)
    distributed_draft_grad = torch.cat(draft_grad_parts, dim=1)

    if rank == 0:
        ref_direct, ref_sparse, ref_layout = _metadata(
            rank=0, size=1, device=device
        )
        ref_context = context_full.detach().clone().requires_grad_(True)
        ref_draft = draft_full.detach().clone().requires_grad_(True)
        ref_positions = (
            _anchor_rope_positions(ref_direct.anchor_positions).unsqueeze(-1)
            + torch.arange(2, device=device).view(1, 1, -1)
        ).flatten(1)
        if kind == "dsa":
            ref_layout = build_dsa_token_layout(
                ref_sparse,
                ref_direct.context_rope_position_ids if _packing_enabled() else ref_direct.context_position_ids,
                ref_positions, rope_dim=64, rope_theta=10000.0,
            )
        reference_output = reference_attention(
            draft_hidden=ref_draft,
            context_hidden=ref_context,
            draft_position_ids=ref_positions,
            context_position_ids=(
                ref_direct.context_rope_position_ids
                if ref_direct.context_rope_position_ids is not None
                else ref_direct.context_position_ids
            ),
            direct_cp_metadata=ref_direct,
            **{f"{kind}_metadata": ref_sparse, f"{kind}_layout": ref_layout},
        )
        output_difference = distributed_output.float() - reference_output.float()
        output_relative_l2 = (
            output_difference.norm()
            / reference_output.float().norm().clamp_min(1e-12)
        )
        print(
            f"OUTPUT rel_l2={output_relative_l2.item():.8f} "
            f"max_abs={output_difference.abs().max().item():.8f}",
            flush=True,
        )
        assert output_relative_l2 < 0.03
        reference_loss = reference_output.float().square().sum()
        reference_loss = (
            reference_loss
            + getattr(reference_attention, f"{kind}_indexer_loss_per_query").float().sum()
        )
        reference_loss.backward()
        for name, actual, expected in (
            ("context", distributed_context_grad, ref_context.grad),
            ("draft", distributed_draft_grad, ref_draft.grad),
        ):
            difference = actual.float() - expected.float()
            relative_l2 = (
                difference.norm() / expected.float().norm().clamp_min(1e-12)
            )
            print(
                f"INPUT_GRAD {name} rel_l2={relative_l2.item():.8f} "
                f"max_abs={difference.abs().max().item():.8f}",
                flush=True,
            )
            assert relative_l2 < (0.03 if kind == "dsa" else 0.15)
        gradient_errors = {}
        for (name, parameter), (_, reference_parameter) in zip(
            attention.named_parameters(),
            reference_attention.named_parameters(),
            strict=True,
        ):
            if parameter.grad is None and reference_parameter.grad is None:
                continue
            difference = (parameter.grad.float() - reference_parameter.grad.float())
            relative_l2 = difference.norm() / reference_parameter.grad.float().norm().clamp_min(1e-12)
            print(
                f"GRAD {name} rel_l2={relative_l2.item():.8f} "
                f"max_abs={difference.abs().max().item():.8f}",
                flush=True,
            )
            gradient_errors[name] = relative_l2.item()
        assert max(gradient_errors.values(), default=0.0) < (0.03 if kind == "dsa" else 0.15), gradient_errors
    dist.barrier()
    dist.destroy_process_group()


def _distributed_csa_parity() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size not in {2, 4}:
        raise RuntimeError("sparse THD parity expects CP2 or CP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20261017)
    stage = os.getenv("ASPEC_THD_STAGE", "sparse")
    attention = DFlashAttention(_csa_config(stage)).to(
        device=device, dtype=torch.bfloat16
    )
    attention.csa_layer.keep_projection_fp32()
    reference_attention = copy.deepcopy(attention) if rank == 0 else None
    context_full = torch.randn(
        1, _context_length(), 256, device=device, dtype=torch.bfloat16
    )
    draft_full = torch.randn(
        1, 8, 256, device=device, dtype=torch.bfloat16
    )
    direct, sparse, layout = _csa_metadata(rank=rank, size=size, device=device)
    local_blocks = direct.local_block_ids.to(device)
    local_context = context_full.index_select(1, direct.context_position_ids[0]).detach()
    local_context.requires_grad_(True)
    local_draft_rows = (
        local_blocks[:, None] * 2 + torch.arange(2, device=device)[None]
    ).flatten()
    local_draft = draft_full.index_select(1, local_draft_rows).detach()
    local_draft.requires_grad_(True)
    draft_positions = (
        _anchor_rope_positions(direct.anchor_positions)
        .index_select(1, local_blocks)
        .unsqueeze(-1)
        + torch.arange(2, device=device).view(1, 1, -1)
    ).flatten(1)
    output = attention(
        draft_hidden=local_draft,
        context_hidden=local_context,
        draft_position_ids=draft_positions,
        context_position_ids=(
            direct.context_rope_position_ids
            if direct.context_rope_position_ids is not None
            else direct.context_position_ids
        ),
        direct_cp_metadata=direct,
        csa_metadata=sparse,
        csa_prepared={4: (layout, None)},
    )
    gathered = [torch.empty_like(output) for _ in range(size)]
    dist.all_gather(gathered, output.detach())
    distributed_output = torch.cat(gathered, dim=1)

    local_loss = output.float().square().sum()
    local_loss = local_loss + attention.csa_indexer_loss_per_query.float().sum()
    local_loss.backward()
    for parameter in attention.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
    context_grad_parts = [torch.empty_like(local_context.grad) for _ in range(size)]
    draft_grad_parts = [torch.empty_like(local_draft.grad) for _ in range(size)]
    dist.all_gather(context_grad_parts, local_context.grad)
    dist.all_gather(draft_grad_parts, local_draft.grad)
    distributed_context_grad = torch.cat(context_grad_parts, dim=1)
    distributed_draft_grad = torch.cat(draft_grad_parts, dim=1)

    if rank == 0:
        ref_direct, ref_sparse, ref_layout = _csa_metadata(
            rank=0, size=1, device=device
        )
        ref_context = context_full.detach().clone().requires_grad_(True)
        ref_draft = draft_full.detach().clone().requires_grad_(True)
        ref_positions = (
            _anchor_rope_positions(ref_direct.anchor_positions).unsqueeze(-1)
            + torch.arange(2, device=device).view(1, 1, -1)
        ).flatten(1)
        reference_output = reference_attention(
            draft_hidden=ref_draft,
            context_hidden=ref_context,
            draft_position_ids=ref_positions,
            context_position_ids=(
                ref_direct.context_rope_position_ids
                if ref_direct.context_rope_position_ids is not None
                else ref_direct.context_position_ids
            ),
            direct_cp_metadata=ref_direct,
            csa_metadata=ref_sparse,
            csa_prepared={4: (ref_layout, None)},
        )
        output_difference = distributed_output.float() - reference_output.float()
        output_relative_l2 = (
            output_difference.norm()
            / reference_output.float().norm().clamp_min(1e-12)
        )
        print(
            f"CSA OUTPUT rel_l2={output_relative_l2.item():.8f} "
            f"max_abs={output_difference.abs().max().item():.8f}",
            flush=True,
        )
        assert output_relative_l2 < 0.03
        reference_loss = reference_output.float().square().sum()
        reference_loss = (
            reference_loss
            + reference_attention.csa_indexer_loss_per_query.float().sum()
        )
        reference_loss.backward()
        for name, actual, expected in (
            ("context", distributed_context_grad, ref_context.grad),
            ("draft", distributed_draft_grad, ref_draft.grad),
        ):
            difference = actual.float() - expected.float()
            relative_l2 = (
                difference.norm() / expected.float().norm().clamp_min(1e-12)
            )
            print(
                f"CSA INPUT_GRAD {name} rel_l2={relative_l2.item():.8f} "
                f"max_abs={difference.abs().max().item():.8f}",
                flush=True,
            )
            assert relative_l2 < 0.15
        gradient_errors = {}
        for (name, parameter), (_, reference_parameter) in zip(
            attention.named_parameters(),
            reference_attention.named_parameters(),
            strict=True,
        ):
            if parameter.grad is None and reference_parameter.grad is None:
                continue
            difference = parameter.grad.float() - reference_parameter.grad.float()
            relative_l2 = (
                difference.norm()
                / reference_parameter.grad.float().norm().clamp_min(1e-12)
            )
            print(
                f"CSA GRAD {name} rel_l2={relative_l2.item():.8f} "
                f"max_abs={difference.abs().max().item():.8f}",
                flush=True,
            )
            gradient_errors[name] = relative_l2.item()
        assert max(gradient_errors.values(), default=0.0) < 0.15, gradient_errors
    dist.barrier()
    dist.destroy_process_group()


def _distributed_mla_parity() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    if size not in {2, 4}:
        raise RuntimeError("MLA THD parity expects CP2 or CP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20261018)
    attention = K3DSparkMLAAttention(_mla_config()).to(
        device=device, dtype=torch.bfloat16
    )
    attention.set_attention_backend("fa4")
    reference_attention = copy.deepcopy(attention) if rank == 0 else None
    context_full = torch.randn(
        1, _context_length(), 256, device=device, dtype=torch.bfloat16
    )
    draft_full = torch.randn(
        1, 8, 256, device=device, dtype=torch.bfloat16
    )
    direct, _, _ = _metadata(rank=rank, size=size, device=device)
    local_blocks = direct.local_block_ids.to(device)
    local_context = context_full.index_select(
        1, direct.context_position_ids[0]
    ).detach().requires_grad_(True)
    local_draft_rows = (
        local_blocks[:, None] * 2 + torch.arange(2, device=device)[None]
    ).flatten()
    local_draft = draft_full.index_select(
        1, local_draft_rows
    ).detach().requires_grad_(True)
    draft_positions = (
        _anchor_rope_positions(direct.anchor_positions)
        .index_select(1, local_blocks)
        .unsqueeze(-1)
        + torch.arange(2, device=device).view(1, 1, -1)
    ).flatten(1)
    output = attention(
        draft_hidden=local_draft,
        context_hidden=local_context,
        draft_position_ids=draft_positions,
        context_position_ids=(
            direct.context_rope_position_ids
            if direct.context_rope_position_ids is not None
            else direct.context_position_ids
        ),
        direct_cp_metadata=direct,
    )
    output_parts = [torch.empty_like(output) for _ in range(size)]
    dist.all_gather(output_parts, output.detach())
    distributed_output = torch.cat(output_parts, dim=1)
    output.float().square().sum().backward()
    for parameter in attention.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
    context_grad_parts = [torch.empty_like(local_context.grad) for _ in range(size)]
    draft_grad_parts = [torch.empty_like(local_draft.grad) for _ in range(size)]
    dist.all_gather(context_grad_parts, local_context.grad)
    dist.all_gather(draft_grad_parts, local_draft.grad)

    if rank == 0:
        ref_direct, _, _ = _metadata(rank=0, size=1, device=device)
        ref_context = context_full.detach().clone().requires_grad_(True)
        ref_draft = draft_full.detach().clone().requires_grad_(True)
        ref_positions = (
            _anchor_rope_positions(ref_direct.anchor_positions).unsqueeze(-1)
            + torch.arange(2, device=device).view(1, 1, -1)
        ).flatten(1)
        reference_output = reference_attention(
            draft_hidden=ref_draft,
            context_hidden=ref_context,
            draft_position_ids=ref_positions,
            context_position_ids=(
                ref_direct.context_rope_position_ids
                if ref_direct.context_rope_position_ids is not None
                else ref_direct.context_position_ids
            ),
            direct_cp_metadata=ref_direct,
        )
        output_difference = distributed_output.float() - reference_output.float()
        output_relative_l2 = (
            output_difference.norm()
            / reference_output.float().norm().clamp_min(1e-12)
        )
        print(f"MLA OUTPUT rel_l2={output_relative_l2.item():.8f}", flush=True)
        assert output_relative_l2 < 0.03
        reference_output.float().square().sum().backward()
        for name, actual, expected in (
            ("context", torch.cat(context_grad_parts, dim=1), ref_context.grad),
            ("draft", torch.cat(draft_grad_parts, dim=1), ref_draft.grad),
        ):
            difference = actual.float() - expected.float()
            relative_l2 = (
                difference.norm() / expected.float().norm().clamp_min(1e-12)
            )
            print(
                f"MLA INPUT_GRAD {name} rel_l2={relative_l2.item():.8f}",
                flush=True,
            )
            assert relative_l2 < 0.15
        gradient_errors = {}
        for (name, parameter), (_, reference_parameter) in zip(
            attention.named_parameters(),
            reference_attention.named_parameters(),
            strict=True,
        ):
            if parameter.grad is None and reference_parameter.grad is None:
                continue
            difference = parameter.grad.float() - reference_parameter.grad.float()
            relative_l2 = (
                difference.norm()
                / reference_parameter.grad.float().norm().clamp_min(1e-12)
            )
            print(
                f"MLA GRAD {name} rel_l2={relative_l2.item():.8f}",
                flush=True,
            )
            gradient_errors[name] = relative_l2.item()
        assert max(gradient_errors.values(), default=0.0) < 0.15
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 and launch with torchrun",
)
def test_qsa_sparse_thd_context_parallel_matches_cp1():
    if not dist.is_initialized():
        pytest.skip("launch this test with torchrun --nproc-per-node=2 or 4")
    _distributed_qsa_parity()


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 and launch with torchrun",
)
def test_csa_sparse_thd_context_parallel_matches_cp1():
    if not dist.is_initialized():
        pytest.skip("launch this test with torchrun --nproc-per-node=2 or 4")
    _distributed_csa_parity()


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_GPU_INTEGRATION=1 and launch with torchrun",
)
def test_mla_thd_context_parallel_matches_cp1():
    if not dist.is_initialized():
        pytest.skip("launch this test with torchrun --nproc-per-node=2 or 4")
    _distributed_mla_parity()


if __name__ == "__main__":
    kind = os.getenv("ASPEC_SPARSE_THD_KIND", "qsa").lower()
    if kind == "csa":
        _distributed_csa_parity()
    elif kind == "dsa":
        _distributed_qsa_parity("dsa")
    elif kind == "mla":
        _distributed_mla_parity()
    else:
        _distributed_qsa_parity()
