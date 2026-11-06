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

# ruff: noqa: I001

"""Opt-in two-GPU correctness check for Direct CP collectives and FA4."""

import os

import pytest

if os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1":
    pytest.skip(
        "set ASPEC_RUN_GPU_INTEGRATION=1 and launch with torchrun",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist

from aspec.models.ops.direct_cp import DirectCPMetadata, direct_cp_attention

def test_cp2_fa4_matches_unsharded_math_forward_and_backward():
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch with torchrun --standalone --nproc-per-node=2")
    pytest.importorskip("flash_attn.cute.interface")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        device = torch.device("cuda", local_rank)
        torch.manual_seed(41)
        shapes = (
            (1, 4, 28, 128),
            (1, 2, 512, 128),
            (1, 2, 512, 128),
            (1, 2, 28, 128),
            (1, 2, 28, 128),
        )
        full = [
            torch.randn(*shape, device=device, dtype=torch.bfloat16)
            for shape in shapes
        ]
        query, context_key, context_value, draft_key, draft_value = full
        query_slice = slice(local_rank * 14, (local_rank + 1) * 14)
        context_slice = slice(local_rank * 256, (local_rank + 1) * 256)
        local_inputs = [
            query[:, :, query_slice].detach().clone().requires_grad_(True),
            context_key[:, :, context_slice].detach().clone().requires_grad_(True),
            context_value[:, :, context_slice].detach().clone().requires_grad_(True),
            draft_key[:, :, query_slice].detach().clone().requires_grad_(True),
            draft_value[:, :, query_slice].detach().clone().requires_grad_(True),
        ]
        reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in full]
        anchors = torch.tensor([[64, 192, 320, 448]], device=device)
        keep = torch.ones_like(anchors, dtype=torch.bool)

        actual_metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=torch.arange(
                context_slice.start,
                context_slice.stop,
                device=device,
            ).view(1, -1),
            block_size=7,
            cp_rank=local_rank,
            cp_size=2,
            group=dist.group.WORLD,
            sequence_length=512,
            backend="fa4",
        )
        reference_metadata = DirectCPMetadata(
            anchor_positions=anchors,
            block_keep_mask=keep,
            context_position_ids=torch.arange(512, device=device).view(1, -1),
            block_size=7,
            cp_rank=0,
            cp_size=1,
            group=None,
            sequence_length=512,
            backend="math",
        )

        actual, actual_lse = direct_cp_attention(*local_inputs, actual_metadata)
        reference, reference_lse = direct_cp_attention(
            *reference_inputs, reference_metadata
        )
        expected = reference[:, :, query_slice]
        expected_lse = reference_lse[:, :, query_slice]
        torch.testing.assert_close(actual, expected, rtol=6e-2, atol=6e-2)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=6e-2, atol=6e-2)

        actual.float().sum().backward()
        reference.float().sum().backward()
        expected_gradients = (
            reference_inputs[0].grad[:, :, query_slice],
            reference_inputs[1].grad[:, :, context_slice],
            reference_inputs[2].grad[:, :, context_slice],
            reference_inputs[3].grad[:, :, query_slice],
            reference_inputs[4].grad[:, :, query_slice],
        )
        for actual_input, expected_gradient in zip(
            local_inputs, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_input.grad,
                expected_gradient,
                rtol=1e-1,
                atol=1e-1,
            )
    finally:
        dist.destroy_process_group()
