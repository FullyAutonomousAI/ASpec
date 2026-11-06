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

"""Opt-in H200 correctness check for native cuDNN THD context parallelism."""

import os

import pytest

if os.getenv("ASPEC_RUN_CUDNN_THD_CP_INTEGRATION") != "1":
    pytest.skip(
        "set ASPEC_RUN_CUDNN_THD_CP_INTEGRATION=1 and launch with torchrun",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist

from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    build_context_positions,
    direct_cp_attention,
)


def test_cudnn_thd_cp_matches_unsharded_math_forward_lse_and_backward(monkeypatch):
    pytest.importorskip("cudnn.flex_attention")
    cp_size = int(os.getenv("WORLD_SIZE", "1"))

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        device = torch.device("cuda", local_rank)
        generator = torch.Generator(device=device).manual_seed(913)
        batch, query_heads, kv_heads = 1, 4, 2
        block_size, num_blocks, context_length = 5, 8, 256
        if num_blocks % cp_size:
            pytest.skip(
                "the representative test geometry requires num_blocks divisible "
                "by the launch CP size"
            )
        query_length = block_size * num_blocks
        anchors = torch.tensor(
            [[17, 49, 81, 113, 145, 177, 209, 241]], device=device
        )
        keep = torch.tensor(
            [[True, True, True, False, True, True, True, True]], device=device
        )
        global_docs = (
            torch.arange(4, device=device).repeat_interleave(64).view(1, -1)
        )
        full = [
            torch.randn(
                shape,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for shape in (
                (batch, query_heads, query_length, 128),
                (batch, kv_heads, context_length, 128),
                (batch, kv_heads, context_length, 128),
                (batch, kv_heads, query_length, 128),
                (batch, kv_heads, query_length, 128),
            )
        ]
        local_blocks = build_context_positions(
            num_blocks, local_rank, cp_size, layout="contiguous"
        ).to(device)
        query_rows = (
            local_blocks[:, None] * block_size
            + torch.arange(block_size, device=device)[None]
        ).flatten()
        context_rows = torch.cat(
            [
                torch.arange(
                    start + 64 * local_rank // cp_size,
                    start + 64 * (local_rank + 1) // cp_size,
                    device=device,
                )
                for start in range(0, context_length, 64)
            ]
        )
        local_inputs = [
            full[0].index_select(2, query_rows).detach().clone().requires_grad_(True),
            full[1].index_select(2, context_rows).detach().clone().requires_grad_(True),
            full[2].index_select(2, context_rows).detach().clone().requires_grad_(True),
            full[3].index_select(2, query_rows).detach().clone().requires_grad_(True),
            full[4].index_select(2, query_rows).detach().clone().requires_grad_(True),
        ]
        reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in full]
        actual_metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=context_rows.view(1, -1),
            context_doc_ids=global_docs.index_select(1, context_rows),
            anchor_doc_ids=global_docs.gather(1, anchors),
            block_size=block_size,
            cp_rank=local_rank,
            cp_size=cp_size,
            group=dist.group.WORLD if cp_size > 1 else None,
            sequence_length=context_length,
            backend="cudnn_flex_attention",
        )
        reference_metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=torch.arange(context_length, device=device).view(1, -1),
            context_doc_ids=global_docs,
            anchor_doc_ids=global_docs.gather(1, anchors),
            block_size=block_size,
            cp_rank=0,
            cp_size=1,
            group=None,
            sequence_length=context_length,
            backend="math",
        )

        actual, actual_lse = direct_cp_attention(*local_inputs, actual_metadata)
        # Numerical control of the old eager BHQD materialization, including
        # downstream CP collectives. Only storage layout differs, not attention.
        import aspec.models.ops.direct_cp as cp_module

        original_partial = cp_module._cudnn_partial
        legacy_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in local_inputs]

        def materialized_partial(*args, **kwargs):
            output, lse = original_partial(*args, **kwargs)
            return output.contiguous(), lse

        with monkeypatch.context() as patch:
            patch.setattr(cp_module, "_cudnn_partial", materialized_partial)
            legacy, legacy_lse = direct_cp_attention(*legacy_inputs, actual_metadata)
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, legacy_lse, rtol=0, atol=0)
        reference, reference_lse = direct_cp_attention(
            *reference_inputs, reference_metadata
        )
        expected = reference.index_select(2, query_rows)
        expected_lse = reference_lse.index_select(2, query_rows)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.04)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.03, rtol=0.04)

        dout = torch.randn(
            reference.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        dlse = torch.randn(
            reference_lse.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        actual_loss = (
            actual.float() * dout.index_select(2, query_rows)
        ).sum() + 0.01 * (
            actual_lse * dlse.index_select(2, query_rows)
        ).sum()
        reference_loss = (reference.float() * dout).sum() + 0.01 * (
            reference_lse * dlse
        ).sum()
        actual_grads = torch.autograd.grad(actual_loss, local_inputs)
        legacy_loss = (
            legacy.float() * dout.index_select(2, query_rows)
        ).sum() + 0.01 * (legacy_lse * dlse.index_select(2, query_rows)).sum()
        legacy_grads = torch.autograd.grad(legacy_loss, legacy_inputs)
        for actual_grad, legacy_grad in zip(actual_grads, legacy_grads, strict=True):
            relative = (actual_grad.float() - legacy_grad.float()).norm() / legacy_grad.float().norm().clamp(min=1e-20)
            assert relative < 1e-3, relative.item()
        reference_grads = torch.autograd.grad(reference_loss, reference_inputs)
        expected_grads = (
            reference_grads[0].index_select(2, query_rows),
            reference_grads[1].index_select(2, context_rows),
            reference_grads[2].index_select(2, context_rows),
            reference_grads[3].index_select(2, query_rows),
            reference_grads[4].index_select(2, query_rows),
        )
        for observed, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(
                observed, expected_grad, atol=0.08, rtol=0.1
            )
        observation = actual_metadata.attention_observation
        assert observation["qkv_format"] == "thd"
        assert observation["kernel_backend"] == "cudnn_flex_attention"
        assert observation["kernel_mode"] == "varlen_interval"
        assert observation["backward_strategy"] == "cudnn_flex_varlen"
        assert observation["backward_observed"] is True
    finally:
        dist.destroy_process_group()
