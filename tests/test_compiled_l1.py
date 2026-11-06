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
from types import SimpleNamespace

import pytest
import torch

from aspec.models.ops.loss import compiled_l1_from_logits


@pytest.mark.parametrize("capability,expected", [((9, 0), True), ((10, 0), False),
                                               ((10, 3), False), ((8, 0), False)])
def test_tiled_l1_selection_preserves_other_architectures(monkeypatch, capability, expected):
    from aspec.models.ops.loss import use_sm90_tiled_l1

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    values = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                             requires_grad=False, shape=(7, 151936),
                             numel=lambda: 7 * 151936, is_contiguous=lambda: True)
    assert use_sm90_tiled_l1(values, values) is expected
    teacher = SimpleNamespace(**{**vars(values), "requires_grad": True})
    assert not use_sm90_tiled_l1(values, teacher)


def test_tiled_l1_selection_keeps_cpu_and_noncompact_reference_path(monkeypatch):
    from aspec.models.ops.loss import use_sm90_tiled_l1

    def unexpected_device_query(_):
        raise AssertionError("ineligible tensors must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_device_query)
    values = torch.zeros(3, 8192, dtype=torch.bfloat16)
    assert not use_sm90_tiled_l1(values, values)
    values = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda:0"),
                             requires_grad=False, shape=(7, 151936),
                             numel=lambda: 7 * 151936, is_contiguous=lambda: False)
    assert not use_sm90_tiled_l1(values, values)


@pytest.mark.parametrize("shape", [(2, 5, 7, 128), (3, 1, 1, 17)])
@pytest.mark.parametrize("target_requires_grad", [False, True])
def test_compiled_l1_preserves_loss_weighting_and_both_gradients(shape, target_requires_grad):
    torch.manual_seed(42)
    draft = torch.randn(shape, requires_grad=True)
    target = torch.randn(shape, requires_grad=target_requires_grad)
    candidate_draft = draft.detach().clone().requires_grad_()
    candidate_target = target.detach().clone().requires_grad_(target_requires_grad)
    weights = torch.rand(shape[:-1])
    weights.flatten()[::3] = 0
    reference = (draft.float().softmax(-1) - target.float().softmax(-1)).abs().sum(-1)
    candidate = compiled_l1_from_logits(candidate_draft, candidate_target)
    assert candidate.shape == shape[:-1]
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)
    (reference * weights).sum().backward()
    (candidate * weights).sum().backward()
    torch.testing.assert_close(candidate_draft.grad, draft.grad, rtol=1e-4, atol=1e-6)
    if target_requires_grad:
        torch.testing.assert_close(candidate_target.grad, target.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1", reason="explicit GPU integration")
def test_compiled_l1_bfloat16_large_vocabulary_gradient():
    torch.manual_seed(42)
    source = torch.randn(1, 16, 7, 248320, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(source)
    weights = torch.rand(source.shape[:-1], device="cuda")
    draft = source.clone().requires_grad_()
    candidate_draft = source.clone().requires_grad_()
    reference = (draft.float().softmax(-1) - target.float().softmax(-1)).abs().sum(-1)
    candidate = compiled_l1_from_logits(candidate_draft, target)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)
    (reference * weights).sum().backward()
    (candidate * weights).sum().backward()
    relative_l2 = (candidate_draft.grad.float() - draft.grad.float()).norm() / draft.grad.float().norm()
    assert relative_l2.item() < 0.003
