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

import os

import pytest
import torch

from aspec.models.draft.csa import _RotaryEmbedding
from aspec.models.draft.llama3_eagle import LlamaLinearScalingRotaryEmbedding, LlamaRotaryEmbedding
from aspec.models.draft.qsa import _PartialRotaryEmbedding

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="requires CUDA and ASPEC_RUN_GPU_INTEGRATION=1",
)


@pytest.mark.parametrize("cls", [_RotaryEmbedding, _PartialRotaryEmbedding])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast", [False, True])
def test_sparse_rotary_gpu_matches_fp32_phase(cls, dtype, autocast):
    torch.manual_seed(107)
    module = cls(64, 40000.0).to(device="cuda", dtype=dtype)
    positions = torch.tensor([[0, 4095, 32767, 131071]], device="cuda")
    x = torch.randn(1, 4, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    freq = 1.0 / (40000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    angles = positions.float().unsqueeze(-1) * freq
    factors = torch.cat((angles, angles), -1).unsqueeze(2)
    expected = ref * factors.cos().to(dtype) + torch.cat(
        (-ref[..., 32:], ref[..., :32]), -1
    ) * factors.sin().to(dtype)
    with torch.autocast("cuda", enabled=autocast, dtype=dtype):
        actual = module(x, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eagle_rotary_cache_growth_gpu_matches_fp32_phase(cls, dtype):
    module = cls(64, max_position_embeddings=64, base=40000.0).to(device="cuda", dtype=dtype)
    module._set_cos_sin_cache(4096, torch.device("cuda"), dtype)
    freq = 1.0 / (40000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    angles = torch.arange(4096, device="cuda", dtype=torch.float32).unsqueeze(-1) * freq
    factors = torch.cat((angles, angles), -1)[None, None]
    torch.testing.assert_close(module.cos_cached, factors.cos().to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(module.sin_cached, factors.sin().to(dtype), rtol=0, atol=0)
