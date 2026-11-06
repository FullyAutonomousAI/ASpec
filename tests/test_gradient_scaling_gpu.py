# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires CUDA gradient normalization integration",
)


@pytest.fixture
def process_group(tmp_path):
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    try:
        yield
    finally:
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize("coefficient", [0.0, 0.00012613054423127323, 1.234567, -0.03125])
def test_scaling_matches_native_rounding_and_mixed_layouts(coefficient):
    from aspec.models.ops.gradient_scaling import scale_gradients_

    torch.manual_seed(92)
    values = [torch.randn(size, device="cuda", dtype=torch.bfloat16)
              for size in (0, 1, 123, 4096, 65537)]
    values.extend((torch.randn(23, 17, device="cuda", dtype=torch.bfloat16).T,
                   torch.randn(31, device="cuda", dtype=torch.float32)))
    actual, reference = [[g.clone() for g in values] for _ in range(2)]
    scale = torch.tensor(coefficient, device="cuda", dtype=torch.float32)
    for gradient in reference:
        gradient.mul_(scale)
    scale_gradients_(actual, scale)
    for left, right in zip(actual, reference, strict=True):
        # Check signed zero and finite bit patterns, not just tolerance equality.
        integer = torch.int16 if left.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(left.view(integer), right.view(integer))


def test_scaling_preserves_nonfinite_values_and_vector_coefficient_promotion():
    from aspec.models.ops.gradient_scaling import scale_gradients_

    values = torch.tensor([float("nan"), float("inf"), -float("inf"),
                           0.0, -0.0, 1e-40, -1e-40, 1.0], device="cuda", dtype=torch.bfloat16)
    for scale in (torch.tensor(0.0, device="cuda"), torch.tensor(0.0123, device="cuda"),
                  torch.tensor([0.0123], device="cuda")):
        actual, reference = values.clone(), values.clone()
        reference.mul_(scale)
        scale_gradients_([actual], scale)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)


def test_dtensor_keeps_its_native_placement_and_scaling(process_group):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate

    from aspec.models.ops.gradient_scaling import scale_gradients_

    mesh = DeviceMesh("cuda", [0])
    values = torch.randn(321, device="cuda", dtype=torch.bfloat16)
    actual = DTensor.from_local(values.clone(), mesh, [Replicate()])
    expected = actual.clone()
    scale = torch.tensor(0.00012613054423127323, device="cuda")
    expected.mul_(scale)
    scale_gradients_([actual], scale)
    assert actual.placements == expected.placements
    assert torch.equal(actual.to_local(), expected.to_local())


def test_normalization_preserves_sixteen_adam_updates_and_restore(process_group):
    from aspec.models.ops.gradient_scaling import scale_gradients_
    from aspec.training.checkpoint import OptimizerState
    from aspec.training.optimizer import BF16Optimizer

    torch.manual_seed(93)
    models = [torch.nn.Linear(257, 129, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())

    def make_optimizer(model):
        return BF16Optimizer(model, lr=6e-4, max_grad_norm=1.0, total_steps=100,
                             warmup_ratio=0.04, weight_decay=0.01)

    optimizers = [make_optimizer(model) for model in models]
    for step in range(16):
        raw = [torch.randn_like(p) * (0.001 if step % 2 else 1000) for p in models[0].parameters()]
        scale = torch.tensor(0.0 if step == 4 else 2 / (951.645751953125 + step), device="cuda")
        for model in models:
            for parameter, gradient in zip(model.parameters(), raw, strict=True):
                parameter.grad = gradient.clone()
        scale_gradients_([p.grad for p in models[0].parameters()], scale)
        for parameter in models[1].parameters():
            parameter.grad.mul_(scale)
        norms = [optimizer.step() for optimizer in optimizers]
        torch.testing.assert_close(norms[0], norms[1], rtol=0, atol=0)
        states = [OptimizerState(model, optimizer).state_dict()
                  for model, optimizer in zip(models, optimizers, strict=True)]
        for key in states[0]["fp32_params"]:
            torch.testing.assert_close(states[0]["fp32_params"][key], states[1]["fp32_params"][key], rtol=0, atol=0)
        for index, fields in states[0]["optim"]["state"].items():
            for key, value in fields.items():
                torch.testing.assert_close(value, states[1]["optim"]["state"][index][key], rtol=0, atol=0)
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            assert torch.equal(left, right)
        if step == 7:
            state = copy.deepcopy(models[0].state_dict())
            moments = copy.deepcopy(states[0])
            scheduler = copy.deepcopy(optimizers[0].scheduler.state_dict())
            models[0] = torch.nn.Linear(257, 129, device="cuda", dtype=torch.bfloat16)
            models[0].load_state_dict(state)
            optimizers[0] = make_optimizer(models[0])
            OptimizerState(models[0], optimizers[0]).load_state_dict(moments)
            optimizers[0].scheduler.load_state_dict(scheduler)
