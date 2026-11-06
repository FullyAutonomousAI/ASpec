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

"""Opt-in native pinned-TorchSpec checkpoint continuation, not an upstream dependency."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tools.validation.dspark_nonpacking_alignment import materialize_reference_optimizer

pytestmark = pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="opt-in real GPU/reference checkpoint validation",
)


def test_native_reference_continuation_keeps_masters_adam_moments_and_lr(tmp_path):
    pytest.importorskip("torchspec")
    from torchspec.training import checkpoint
    from torchspec.training.optimizer import BF16Optimizer

    assert not dist.is_initialized()
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:

        def actor():
            model = torch.nn.Linear(4, 2).to("cuda", dtype=torch.bfloat16)
            optimizer = BF16Optimizer(
                model,
                lr=6e-4,
                min_lr=0.0,
                weight_decay=0.0,
                max_grad_norm=1.0,
                total_steps=1180,
                warmup_ratio=0.04,
            )
            return SimpleNamespace(
                model=model,
                optimizer=optimizer,
                lr_scheduler=optimizer.lr_scheduler,
                global_step=0,
                args=SimpleNamespace(
                    checkpoint_dir=str(tmp_path / "reference"),
                    load_path=None,
                    ckpt_step=None,
                    continual_training=False,
                ),
            )

        def update(value):
            value.model(
                torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
            ).float().square().mean().backward()
            value.optimizer.step()
            value.global_step += 1

        source = actor()
        for _ in range(3):
            update(source)
        checkpoint.save(source, 2)
        unprimed = actor()
        unprimed.args.load_path = source.args.checkpoint_dir
        checkpoint.load(unprimed)
        print("NATIVE_UNPRIMED_ADAM_SLOTS", len(unprimed.optimizer.state))
        restored = actor()
        restored.args.load_path = source.args.checkpoint_dir
        directory = tmp_path / "reference/iter_0000003/optimizer"
        assert not restored.optimizer.state
        assert materialize_reference_optimizer(restored.optimizer, directory) == 2
        payload = checkpoint.load(restored)
        checkpoint.finalize_load(restored, payload)
        assert restored.global_step == 3
        for actual, expected in zip(
            restored.optimizer.fp32_params, source.optimizer.fp32_params, strict=True
        ):
            assert torch.equal(actual, expected)
        for actual, expected in zip(
            restored.optimizer.state.values(), source.optimizer.state.values(), strict=True
        ):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(actual[key], expected[key]), key
        assert restored.optimizer.get_learning_rate() == source.optimizer.get_learning_rate()
        update(source)
        update(restored)
        for actual, expected in zip(
            restored.optimizer.fp32_params, source.optimizer.fp32_params, strict=True
        ):
            assert torch.equal(actual, expected)
    finally:
        dist.destroy_process_group()


def test_native_optimizers_agree_with_identical_bf16_gradients():
    pytest.importorskip("torchspec")
    from torchspec.training.optimizer import BF16Optimizer as TorchOptimizer

    from aspec.training.optimizer import BF16Optimizer as FastOptimizer

    models = [
        torch.nn.Linear(1024, 1024, bias=False).to("cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [
        kind(
            model,
            lr=6e-4,
            min_lr=0.0,
            weight_decay=0.0,
            max_grad_norm=1.0,
            total_steps=1180,
            warmup_ratio=0.04,
        )
        for model, kind in zip(models, (FastOptimizer, TorchOptimizer), strict=True)
    ]
    generator = torch.Generator(device="cuda").manual_seed(737)
    for _ in range(32):
        gradient = (
            torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, generator=generator)
            * 0.001
        )
        for model, optimizer in zip(models, optimizers, strict=True):
            model.weight.grad = gradient.clone()
            optimizer.step()
        assert optimizers[0].get_learning_rate() == optimizers[1].get_learning_rate()
    torch.testing.assert_close(
        optimizers[0].fp32_params[0], optimizers[1].fp32_params[0], atol=1e-7, rtol=1e-6
    )
