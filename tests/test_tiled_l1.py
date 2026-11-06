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

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires the CUDA loss-kernel integration environment",
)


@pytest.mark.parametrize("vocab", [511, 4097, 151936])
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_tiled_l1_matches_fp64_oracle_and_bf16_gradient(vocab, scale):
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    generator = torch.Generator(device="cuda").manual_seed(872)
    draft = torch.randn(2, 3, vocab, device="cuda", dtype=torch.bfloat16,
                        generator=generator) * scale
    target = torch.randn(draft.shape, device="cuda", dtype=torch.bfloat16,
                         generator=generator) * scale
    upstream = torch.randn(2, 3, device="cuda", generator=generator)
    actual_input = draft.clone().requires_grad_()
    actual = tiled_l1_from_logits(actual_input, target)
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    reference_input = draft.clone().requires_grad_()
    reference = (reference_input.double().softmax(-1) - target.double().softmax(-1)).abs().sum(-1)
    reference_grad = torch.autograd.grad(reference, reference_input, upstream.double())[0]
    torch.testing.assert_close(actual.double(), reference, atol=2e-6, rtol=2e-6)
    relative = (actual_grad.float() - reference_grad.float()).norm() / reference_grad.float().norm()
    assert relative.item() < 0.01
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-7, rtol=0.02)


def test_identical_distributions_have_zero_loss_and_gradient():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    draft = torch.randn(5, 4103, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = tiled_l1_from_logits(draft, draft.detach())
    gradient = torch.autograd.grad(loss.sum(), draft)[0]
    assert torch.count_nonzero(loss) == 0
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.skipif(os.getenv("ASPEC_RUN_LARGE_GPU_TESTS") != "1",
                    reason="opt-in >2^31-element pointer-offset regression (about 14 GB)")
def test_tiled_l1_offsets_beyond_signed_int32_match_last_row_oracle():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        pytest.skip("large-offset regression needs 16 GiB of free device memory")
    vocab = 151936
    rows = (2**31) // vocab + 2
    assert (rows - 1) * vocab > 2**31
    draft = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
    teacher = torch.zeros_like(draft)
    generator = torch.Generator(device="cuda").manual_seed(872)
    last = torch.randn(vocab, device="cuda", dtype=torch.bfloat16, generator=generator)
    last_teacher = torch.randn(vocab, device="cuda", dtype=torch.bfloat16, generator=generator)
    draft[-1].copy_(last)
    teacher[-1].copy_(last_teacher)
    draft.requires_grad_()
    actual = tiled_l1_from_logits(draft, teacher)
    actual.sum().backward()
    reference_input = last.double().requires_grad_()
    reference = (reference_input.softmax(-1) - last_teacher.double().softmax(-1)).abs().sum()
    reference.backward()
    torch.testing.assert_close(actual[-1].double(), reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(draft.grad[-1], reference_input.grad.bfloat16(), atol=1e-7, rtol=0.02)
    assert actual[0] == 0
    assert not draft.grad[0].count_nonzero()


def test_tiled_l1_rejects_a_trainable_teacher():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    values = torch.randn(3, 31, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(ValueError, match="frozen teacher"):
        tiled_l1_from_logits(values, values)


@pytest.mark.parametrize("side", ["draft", "teacher"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_tiled_l1_preserves_nonfinite_reference_gradients(side, value):
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    draft = torch.randn(2, 4103, device="cuda", dtype=torch.bfloat16)
    teacher = torch.randn_like(draft)
    (draft if side == "draft" else teacher)[0, 7] = value
    actual_input = draft.clone().requires_grad_()
    reference_input = draft.clone().requires_grad_()
    actual = tiled_l1_from_logits(actual_input, teacher)
    reference = (reference_input.float().softmax(-1) - teacher.float().softmax(-1)).abs().sum(-1)
    actual_grad = torch.autograd.grad(actual.sum(), actual_input)[0]
    reference_grad = torch.autograd.grad(reference.sum(), reference_input)[0]
    assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
    assert torch.equal(torch.isfinite(actual_grad), torch.isfinite(reference_grad))
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0.02, atol=1e-6, equal_nan=True)
    torch.testing.assert_close(actual[1], reference[1], rtol=2e-6, atol=2e-6)


def test_tiled_l1_rejects_invalid_kernel_inputs():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    values = torch.randn(3, 31, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA logits"):
        tiled_l1_from_logits(values.cpu(), values.cpu())
    with pytest.raises(ValueError, match="BF16"):
        tiled_l1_from_logits(values.float(), values.float())
    with pytest.raises(ValueError, match="nonempty"):
        tiled_l1_from_logits(values[:0], values[:0])
    with pytest.raises(ValueError, match="contiguous"):
        tiled_l1_from_logits(values.T, values.T)


@pytest.mark.parametrize("all_masked", [False, True])
def test_masked_vocabulary_tiles_preserve_global_softmax(all_masked):
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    draft = torch.randn(2, 8201, device="cuda", dtype=torch.bfloat16)
    teacher = torch.randn_like(draft)
    draft[:, :4096] = -float("inf")
    teacher[:, 4096:8192] = -float("inf")
    if all_masked:
        draft[0] = -float("inf")
        teacher[1] = -float("inf")
    actual_input = draft.clone().requires_grad_()
    reference_input = draft.clone().requires_grad_()
    actual = tiled_l1_from_logits(actual_input, teacher)
    reference = (reference_input.float().softmax(-1) - teacher.float().softmax(-1)).abs().sum(-1)
    actual_grad = torch.autograd.grad(actual.sum(), actual_input)[0]
    reference_grad = torch.autograd.grad(reference.sum(), reference_input)[0]
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=0.02, equal_nan=True)
