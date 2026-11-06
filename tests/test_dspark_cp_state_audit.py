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

from tools.validation.dspark_cp_state_audit import (
    NativeUpdateAudit,
    adamw_oracle,
    add_statistics,
    finish_statistics,
    load_snapshot,
    named_trainables,
)


def test_analytical_adamw_matches_16_real_updates_including_warmup_and_decay():
    parameter = torch.nn.Parameter(torch.tensor([0.03, -0.2, 0.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, weight_decay=0.1, eps=1e-8)
    for step in range(1, 17):
        optimizer.param_groups[0]["lr"] = min(step - 1, 4) * 0.00015
        previous = parameter.detach().clone()
        state = optimizer.state[parameter]
        first = state["exp_avg"].clone() if state else torch.zeros_like(previous)
        second = state["exp_avg_sq"].clone() if state else torch.zeros_like(previous)
        gradient = torch.tensor([(-1) ** step * 0.3, 1e-7, 0.0], dtype=torch.float64)
        parameter.grad = gradient
        expected = adamw_oracle(previous, first, second, gradient, step, optimizer.param_groups[0])
        optimizer.step()
        for field, actual in {
            "master": parameter,
            **{key: optimizer.state[parameter][key] for key in ("exp_avg", "exp_avg_sq")},
        }.items():
            torch.testing.assert_close(actual, expected[field], rtol=1e-13, atol=1e-15)


def test_full_coordinate_statistics_do_not_hide_update_error_in_large_weights():
    total = torch.zeros(5, dtype=torch.float64)
    add_statistics(total, torch.tensor([3.0, 4.0]), torch.tensor([3.0, 5.0]))
    row = finish_statistics(total.tolist())
    assert row["relative_l2"] == pytest.approx(0.2)
    assert row["difference_l2"] == row["max_absolute"] == 1
    assert row["coordinates"] == 2


def test_state_restore_is_exact_and_does_not_change_lr_or_moments():
    model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    masters = [
        parameter.detach().float().clone().requires_grad_() for parameter in model.parameters()
    ]
    adam = torch.optim.AdamW(masters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(adam, lambda step: 1.0)
    optimizer = SimpleNamespace(
        model_params=list(model.parameters()),
        fp32_params=masters,
        optimizer=adam,
        scheduler=scheduler,
        get_learning_rate=lambda: adam.param_groups[0]["lr"],
    )
    source = {
        name: {
            "model": torch.full_like(parameter, 0.5),
            "master": torch.full_like(master, 0.5001),
            "exp_avg": torch.full_like(master, 0.1),
            "exp_avg_sq": torch.full_like(master, 0.01),
            "optimizer_step": 15.0,
        }
        for (name, parameter), master in zip(model.named_parameters(), masters, strict=True)
    }
    payload = {
        "parameters": source,
        "optimizer_group": {
            key: value for key, value in adam.param_groups[0].items() if key != "params"
        },
        "scheduler": scheduler.state_dict(),
        "learning_rate": optimizer.get_learning_rate(),
    }
    load_snapshot(model, optimizer, payload)
    for (name, parameter), master in zip(model.named_parameters(), masters, strict=True):
        assert torch.equal(parameter, source[name]["model"])
        assert torch.equal(master, source[name]["master"])
        assert float(adam.state[master]["step"]) == 15
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(adam.state[master][key], source[name][key])
    assert optimizer.get_learning_rate() == 0.001
    with pytest.raises(ValueError, match="parameter sets differ"):
        load_snapshot(model, optimizer, dict(payload, parameters={}))
    with pytest.raises(ValueError, match="contract changed"):
        load_snapshot(
            model,
            optimizer,
            dict(payload, optimizer_group=dict(payload["optimizer_group"], eps=0.1)),
        )


def test_audit_rejects_model_master_order_mismatch():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="ordering differs"):
        named_trainables(
            model, SimpleNamespace(model_params=list(reversed(list(model.parameters()))))
        )


@pytest.mark.skipif(
    os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1",
    reason="Opt-in native CUDA/FSDP observer check",
)
def test_full_update_observation_preserves_all_16_native_states(tmp_path):
    import torch.distributed as dist

    from aspec.training.optimizer import BF16Optimizer

    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
        )
    torch.manual_seed(17)
    models = [torch.nn.Linear(7, 3, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [
        BF16Optimizer(
            model,
            lr=6e-4,
            min_lr=6e-5,
            weight_decay=0.1,
            max_grad_norm=0.5,
            total_steps=100,
            warmup_ratio=0.04,
        )
        for model in models
    ]
    audit = NativeUpdateAudit(models[1], optimizers[1], tmp_path, 1, [2, 16])
    for step in range(1, 17):
        gradients = [torch.randn_like(parameter) for parameter in models[0].parameters()]
        for model in models:
            for parameter, gradient in zip(model.parameters(), gradients, strict=True):
                parameter.grad = gradient.clone()
        audit.before(step)
        first_norm, second_norm = [float(optimizer.step()) for optimizer in optimizers]
        audit.after(step, second_norm)
        assert first_norm == second_norm
        for left, right in zip(optimizers[0].fp32_params, optimizers[1].fp32_params, strict=True):
            assert torch.equal(left, right)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(
                    optimizers[0].optimizer.state[left][key],
                    optimizers[1].optimizer.state[right][key],
                )
    assert {path.name for path in tmp_path.glob("*.pt")} == {
        "before-0002.pt",
        "after-0002.pt",
        "before-0016.pt",
        "after-0016.pt",
    }
    dist.destroy_process_group()
