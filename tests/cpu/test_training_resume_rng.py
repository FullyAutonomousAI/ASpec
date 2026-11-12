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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from aspec.training import checkpoint, dflash_trainer
from aspec.training.dspark_trainer import DSparkTrainer


@pytest.mark.parametrize("trainer_class,anchor_offset", [
    (dflash_trainer.DFlashTrainer, 1), (DSparkTrainer, 0),
])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_training_rng_is_restored_after_target_helper_initialization(
    monkeypatch, trainer_class, anchor_offset, explicit_none,
):
    saved_rng = torch.get_rng_state().clone()

    class Draft(torch.nn.Linear):
        layers = []

        def load_embedding(self, *_args, **_kwargs):
            pass

        def freeze_embedding(self):
            pass

        def keep_csa_projection_precision(self):
            pass

    # Legacy Draft/wrapper extensions deliberately have no MTP-specific methods
    # or attributes. Disabled optional features must not require those hooks.
    trainer = object.__new__(trainer_class)
    trainer._anchor_slot_offset = anchor_offset
    trainer.args = SimpleNamespace(
        lr_total_steps=32, learning_rate=0.001, max_grad_norm=1, continual_training=False
    )
    trainer.dp_rank, trainer.global_step = 0, 0
    trainer.dp_mesh, trainer.fsdp_cpu_offload = None, False
    trainer.prof = SimpleNamespace(on_init_end=Mock())
    config = dict(num_target_layers=1, target_hidden_size=2, target_num_hidden_layers=2)
    if explicit_none:
        config["mtp_conditioning"] = None
    trainer._draft_config_class = lambda **_: SimpleNamespace(**config)
    trainer._get_init_weight_context_manager = lambda: nullcontext
    trainer._build_draft_model = lambda _: Draft(2, 2)

    def wrapper(draft):
        model = torch.nn.Module()
        model.draft_model = draft
        return model

    trainer._build_training_wrapper = wrapper

    def initialize_target(_path):
        torch.rand(123)  # A helper's temporary initialization must not advance restored RNG.
        trainer.target_lm_head = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(2, 2)))

    trainer._init_target_lm_head = initialize_target
    monkeypatch.setattr(dflash_trainer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dflash_trainer.dist, "barrier", lambda **_: None)
    monkeypatch.setattr(dflash_trainer, "get_gloo_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(dflash_trainer, "apply_fsdp2", lambda model, **_: model)
    monkeypatch.setattr(
        dflash_trainer, "fsdp2_load_full_state_dict", lambda model, *_a, **_k: model
    )
    monkeypatch.setattr(
        dflash_trainer, "BF16Optimizer", lambda *_a, **_k: SimpleNamespace(lr_scheduler=object())
    )
    monkeypatch.setattr(
        checkpoint,
        "load",
        lambda _: {
            "rng": {"torch": saved_rng, "cuda": None},
            "metadata": {"global_step": 16, "next_step": 17},
        },
    )
    trainer.init_model({}, "unused")
    assert trainer.global_step == 16
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert trainer._anchor_slot_offset == anchor_offset
    assert not trainer._mtp_binding_validated
