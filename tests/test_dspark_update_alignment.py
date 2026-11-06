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

"""Update diagnosis must observe, not alter, optimizer arithmetic."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.validation.dspark_fa4_repeatability import install_deterministic_fa4, install_fp32_ce
from tools.validation.dspark_state_transplant import install_transplant
from tools.validation.dspark_update_alignment import install_update_observer, save_new
from tools.validation.dspark_update_replay import add_difference, finish_difference, new_difference


class TinyOptimizer:
    def __init__(self, model):
        self.model_params = list(model.parameters())
        self.fp32_params = [p.detach().float().clone().requires_grad_() for p in self.model_params]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: step / 16)
        self.max_grad_norm = 1.0

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        for model, master in zip(self.model_params, self.fp32_params, strict=True):
            master.grad = model.grad.float().clone()
        norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for model, master in zip(self.model_params, self.fp32_params, strict=True):
                model.copy_(master)
                model.grad = None
        return norm


def test_update_observer_leaves_all_16_optimizer_states_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    torch.manual_seed(17)
    models = [torch.nn.Linear(7, 3, dtype=torch.bfloat16) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    trainers = [
        SimpleNamespace(model=model, optimizer=TinyOptimizer(model), global_step=0)
        for model in models
    ]
    install_update_observer(trainers[1], tmp_path)
    for step in range(16):
        gradients = [torch.randn_like(parameter) for parameter in models[0].parameters()]
        for trainer in trainers:
            trainer.global_step = step
            for parameter, gradient in zip(trainer.model.parameters(), gradients, strict=True):
                parameter.grad = gradient.clone()
            trainer.optimizer.step()
        for left, right in zip(
            trainers[0].optimizer.fp32_params, trainers[1].optimizer.fp32_params, strict=True
        ):
            assert torch.equal(left, right)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(
                    trainers[0].optimizer.optimizer.state[left][key],
                    trainers[1].optimizer.optimizer.state[right][key],
                )
    records = [
        json.loads(line) for line in (tmp_path / "update-witnesses.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == list(range(1, 17))
    assert records[0]["learning_rate_used"] == 0
    assert {p.name for p in tmp_path.glob("*.pt")} == {
        "gradients-0001.pt",
        "gradients-0002.pt",
        "state-after-0002.pt",
    }
    state = torch.load(tmp_path / "state-after-0002.pt", weights_only=True)
    assert all(value["optimizer_step"] == 2 for value in state.values())


def test_save_rejects_existing_or_incomplete_capture(tmp_path):
    path = tmp_path / "snapshot.pt"
    save_new({"value": torch.ones(2)}, path)
    with pytest.raises(FileExistsError):
        save_new({}, path)
    assert torch.equal(torch.load(path, weights_only=True)["value"], torch.ones(2))
    incomplete = tmp_path / "other.pt.partial"
    incomplete.touch()
    with pytest.raises(FileExistsError):
        save_new({}, tmp_path / "other.pt")


def test_complete_vector_difference_and_update_metric():
    total = new_difference()
    add_difference(total, torch.tensor([3.0, 4.0]), torch.tensor([3.0, 5.0]))
    result = finish_difference(total)
    assert result["relative_l2"] == pytest.approx(0.2)
    assert result["max_absolute"] == 1
    assert result["different_fraction"] == 0.5


def test_dense_deterministic_diagnosis_keeps_forward_and_document_mask(monkeypatch):
    captured = {}

    def original(*args, **kwargs):
        captured.update(kwargs)
        return "observed"

    root = ModuleType("flash_attn")
    cute = ModuleType("flash_attn.cute")
    interface = ModuleType("flash_attn.cute.interface")
    root.cute, cute.interface = cute, interface
    interface.flash_attn_func = original
    for name, module in (
        ("flash_attn", root),
        ("flash_attn.cute", cute),
        ("flash_attn.cute.interface", interface),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    assert install_deterministic_fa4() is original
    forward, backward, mask = object(), object(), object()
    assert (
        interface.flash_attn_func(
            None, block_sparse_tensors=forward, block_sparse_tensors_bwd=backward, mask_mod=mask
        )
        == "observed"
    )
    assert captured["deterministic"] is True
    assert captured["block_sparse_tensors_bwd"] is None
    assert captured["block_sparse_tensors"] is forward
    assert captured["mask_mod"] is mask


def test_real_state_transplant_occurs_once_without_changing_lr(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    model = torch.nn.Linear(7, 3, dtype=torch.bfloat16)
    optimizer = TinyOptimizer(model)
    for _ in range(2):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    trainer = SimpleNamespace(
        model=model, optimizer=optimizer, global_step=2, _train_step=lambda: "original"
    )
    source = {}
    for (name, parameter), master in zip(
        model.named_parameters(), optimizer.fp32_params, strict=True
    ):
        source[name] = {
            "model": torch.full_like(parameter, 7),
            "master": torch.full_like(master, 8),
            "exp_avg": torch.full_like(master, 9),
            "exp_avg_sq": torch.full_like(master, 10),
            "optimizer_step": 2,
        }
    path = tmp_path / "source.pt"
    save_new(source, path)
    learning_rate = optimizer.get_learning_rate()
    install_transplant(trainer, path, tmp_path)
    assert trainer._train_step() == "original"
    for parameter, master in zip(model.parameters(), optimizer.fp32_params, strict=True):
        assert torch.equal(parameter, torch.full_like(parameter, 7))
        assert torch.equal(master, torch.full_like(master, 8))
        assert torch.equal(optimizer.optimizer.state[master]["exp_avg"], torch.full_like(master, 9))
        with torch.no_grad():
            master.fill_(11)
    assert trainer._train_step() == "original"
    assert all(torch.equal(master, torch.full_like(master, 11)) for master in optimizer.fp32_params)
    assert optimizer.get_learning_rate() == learning_rate
    assert json.loads((tmp_path / "transplant.json").read_text())["transplant_before_step"] == 3


def test_ce_precision_diagnosis_is_scoped_and_preserves_bf16_projection(monkeypatch):
    import torch.nn.functional as functional

    from aspec.models import dspark

    monkeypatch.setattr(dspark, "F", functional)
    original_ce = functional.cross_entropy
    assert install_fp32_ce() is functional
    logits = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([1])
    actual = dspark.F.cross_entropy(logits, target, reduction="none")
    expected = functional.cross_entropy(logits.float(), target, reduction="none")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert logits.grad.dtype == torch.bfloat16
    assert functional.cross_entropy is original_ce
    assert dspark.F.linear is functional.linear
    assert (
        dspark.F.linear(logits.detach(), torch.ones(2, 3, dtype=torch.bfloat16)).dtype
        == torch.bfloat16
    )
