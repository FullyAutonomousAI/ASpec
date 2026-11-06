# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires the CUDA loss-kernel integration environment",
)


@pytest.mark.parametrize("vocab", [511, 4097, 8192, 151936])
@pytest.mark.parametrize("scale", [1, 10])
def test_indexed_l1_matches_gathered_loss_and_gradient_bitwise(vocab, scale):
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    generator = torch.Generator(device="cuda").manual_seed(872)
    data = torch.randn(2, 3, vocab, device="cuda", dtype=torch.bfloat16, generator=generator) * scale
    teacher = torch.randn(19, vocab, device="cuda", dtype=torch.bfloat16, generator=generator) * scale
    ids = torch.tensor([18, 0, 4, 4, 0, 18], device="cuda")
    actual_input, reference_input = data.clone().requires_grad_(), data.clone().requires_grad_()
    actual = tiled_l1_from_logits(actual_input, teacher, target_indices=ids)
    reference = tiled_l1_from_logits(reference_input, teacher[ids].reshape_as(data))
    upstream = torch.randn(3, 2, device="cuda", generator=generator).T * 0.017
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    reference_grad = torch.autograd.grad(reference, reference_input, upstream)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)


def test_indexed_identical_sharp_distributions_have_zero_loss_and_gradient():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    teacher = torch.randn(4, 151936, device="cuda", dtype=torch.bfloat16) * 100
    ids = torch.tensor([3, 0, 3, 2], device="cuda")
    data = teacher[ids].requires_grad_()
    loss = tiled_l1_from_logits(data, teacher, target_indices=ids)
    gradient = torch.autograd.grad(loss.sum(), data)[0]
    assert not loss.count_nonzero()
    assert not gradient.count_nonzero()


@pytest.mark.parametrize("side", ["draft", "teacher"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf"), "masked"])
def test_indexed_nonfinite_and_masked_rows_match_fp32_reference(side, nonfinite):
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    data = torch.randn(3, 8201, device="cuda", dtype=torch.bfloat16)
    teacher = torch.randn_like(data)
    data[:, :4096] = -float("inf")
    teacher[:, 4096:8192] = -float("inf")
    modified = data if side == "draft" else teacher
    if nonfinite == "masked":
        modified[2] = -float("inf")
    else:
        modified[2, 7] = nonfinite
    ids = torch.tensor([2, 1, 2], device="cuda")
    actual_input, reference_input = data.clone().requires_grad_(), data.clone().requires_grad_()
    actual = tiled_l1_from_logits(actual_input, teacher, target_indices=ids)
    reference = (reference_input.float().softmax(-1) - teacher[ids].float().softmax(-1)).abs().sum(-1)
    actual_grad = torch.autograd.grad(actual.sum(), actual_input)[0]
    reference_grad = torch.autograd.grad(reference.sum(), reference_input)[0]
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=0.02, equal_nan=True)


def test_indexed_teacher_rejects_mismatched_geometry_dtype_and_trainable_table():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    data = torch.randn(3, 8192, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(3, device="cuda")
    for teacher, indices in ((data, ids.int()), (data, ids[:2]), (data[:, :0].contiguous(), ids),
                             (data[None], ids), (data, ids.cpu())):
        with pytest.raises(ValueError, match="indexed teacher"):
            tiled_l1_from_logits(data, teacher, target_indices=indices)
    with pytest.raises(ValueError, match="frozen teacher"):
        tiled_l1_from_logits(data, data.requires_grad_(), target_indices=ids)


def test_indexed_bounds_do_not_compile_a_kernel_per_context_length():
    from aspec.models.ops.tiled_l1 import _stats

    data = torch.zeros(2, 8192, device="cuda", dtype=torch.bfloat16)
    teacher = torch.zeros(129, 8192, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(2, device="cuda", dtype=torch.int64)
    stats = torch.empty(2, 2, 4, device="cuda", dtype=torch.float32)
    kernels = [_stats.warmup(data, teacher, stats, 8192, 2, 4096, ids, count,
                              grid=(2, 2), debug=True, enable_fp_fusion=False)
               for count in (128, 127, 123)]
    assert len({kernel.hash for kernel in kernels}) == 1


def test_indexed_bounds_fail_closed_without_poisoning_parent_context():
    result = subprocess.run([
        sys.executable, "-c",
        "import torch; from aspec.models.ops.tiled_l1 import tiled_l1_from_logits; "
        "x=torch.zeros(2,8192,device='cuda',dtype=torch.bfloat16); "
        "ids=torch.tensor([0,9],device='cuda'); "
        "tiled_l1_from_logits(x,x,target_indices=ids); torch.cuda.synchronize()",
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "L1 teacher row index out of bounds" in result.stderr


@pytest.mark.parametrize("batch", [1, 2])
def test_dspark_indexed_head_preserves_batch_ownership_and_trainable_heads(batch):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    torch.manual_seed(37)
    config = DSparkConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                         vocab_size=8192, mask_token_id=8191, markov_rank=16,
                         num_target_layers=1, target_hidden_size=64,
                         target_num_hidden_layers=4, target_layer_ids=[1])
    actual = DSparkModel(DSparkDraftModel(config)).cuda().bfloat16()
    reference = copy.deepcopy(actual)
    reference._use_l1_kernel_optimizations = False
    hidden = torch.randn(batch, 8, 7, 64, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(8192, 64, device="cuda", dtype=torch.bfloat16) * 0.03
    teacher_hidden = torch.randn(batch, 97, 64, device="cuda", dtype=torch.bfloat16)
    teacher = torch.nn.functional.linear(teacher_hidden, head)
    positions = torch.randint(97, (batch, 8, 7), device="cuda")
    positions[:, :2] = 0
    labels = torch.randint(8192, positions.shape, device="cuda")
    mask = (torch.rand(positions.shape, device="cuda") > 0.2).float()
    weights = mask * torch.exp(-torch.arange(7, device="cuda").float() / 4)
    outputs, gradients = [], []
    for model in (actual, reference):
        values = hidden.clone().requires_grad_()
        terms = model._prediction_terms(values, labels, labels, head,
                                         torch.zeros_like(values), mask, weights,
                                         cached_target_logits=teacher, cached_target_positions=positions)
        ((0.1 * terms[0] + 0.9 * terms[1] + terms[2]) * 0.017).backward()
        outputs.append(terms)
        gradients.append([values.grad] + [p.grad for p in model.parameters() if p.grad is not None])
    assert actual.l1_kernel_invocations == {"indexed_tiled_fp32": 1}
    assert reference.l1_kernel_invocations == {"tiled_fp32": 1}
    for candidate, old in zip(*outputs, strict=True):
        torch.testing.assert_close(candidate, old, atol=0, rtol=0)
    for candidate, old in zip(*gradients, strict=True):
        torch.testing.assert_close(candidate, old, atol=0, rtol=0)


@pytest.mark.skipif(os.getenv("ASPEC_RUN_LARGE_GPU_TESTS") != "1", reason="opt-in 4 GiB teacher table")
def test_indexed_teacher_offsets_widen_before_vocab_stride():
    from aspec.models.ops.tiled_l1 import tiled_l1_from_logits

    vocab = 151936
    rows = 2**31 // vocab + 2
    teacher = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
    teacher[-1].normal_()
    data = torch.randn(2, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([rows - 1, 0], device="cuda")
    assert ids[0] * vocab > 2**31
    actual = tiled_l1_from_logits(data, teacher, target_indices=ids)
    actual_grad = torch.autograd.grad(actual.sum(), data)[0]
    reference_input = data.detach().clone().requires_grad_()
    reference = tiled_l1_from_logits(reference_input, teacher[ids])
    reference_grad = torch.autograd.grad(reference.sum(), reference_input)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)
