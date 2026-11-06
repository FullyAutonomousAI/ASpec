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

"""EAGLE3 cached-Target-logit correctness and runtime selection."""

from argparse import Namespace

import pytest
import torch

from aspec.models import eagle3
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.training.eagle3_trainer import Eagle3Trainer


def _draft():
    config = AutoDraftModelConfig.from_dict(
        {
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "hidden_size": 16,
            "target_hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "draft_vocab_size": 32,
            "num_aux_hidden_states": 5,
            "target_layer_ids": [1, 3, 5, 7, 9],
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "pad_token_id": 0,
            "tie_word_embeddings": False,
        }
    )
    return AutoEagle3DraftModel.from_config(config)


@pytest.fixture
def eager_losses(monkeypatch):
    # The contract under test is the loss math, not compiler startup.
    for name in dir(eagle3):
        fn = getattr(eagle3, name)
        if name.startswith("compiled_"):
            monkeypatch.setattr(
                eagle3, name, getattr(fn, "_torchdynamo_orig_callable", fn)
            )


@pytest.mark.parametrize("loss_type", ["forward_kl", "lk_alpha", "lk_lambda"])
def test_cached_logits_match_lazy_target_outputs_and_gradients(loss_type, eager_losses):
    torch.manual_seed(17)
    draft = _draft()
    model = eagle3.Eagle3Model(draft, length=3, loss_type=loss_type)
    teacher = torch.randn(2, 11, 16)
    target_head = torch.randn(32, 16)
    lazy = eagle3.compute_lazy_target_padded(teacher, target_head, 3)
    cached = eagle3.compute_cached_target_logits_padded(teacher, target_head, 3)
    mask = torch.randint(0, 2, (2, 11)).float()
    states = torch.randn(2, 11, 16, requires_grad=True)
    norm, head, eps = draft.get_lm_head_params()

    expected = model._calculate_loss(
        states, lazy, mask, 1, 11, norm, head, eps, None
    )
    expected_grad = torch.autograd.grad(expected[0], (states, norm, head))
    actual = model._calculate_loss(
        states, cached, mask, 1, 11, norm, head, eps, None
    )
    actual_grad = torch.autograd.grad(actual[0], (states, norm, head))

    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("loss_type", ["forward_kl", "lk_alpha", "lk_lambda"])
def test_disabling_loss_checkpoint_preserves_outputs_and_gradients(loss_type, eager_losses):
    torch.manual_seed(23)
    teacher = torch.randn(2, 11, 16)
    target_head = torch.randn(32, 16)
    target = eagle3.compute_cached_target_logits_padded(teacher, target_head, 3)
    mask = torch.randint(0, 2, (2, 11)).float()

    def run(checkpoint_loss):
        draft = _draft()
        model = eagle3.Eagle3Model(
            draft,
            length=3,
            gradient_checkpointing=True,
            loss_gradient_checkpointing=checkpoint_loss,
            loss_type=loss_type,
        )
        model.train()
        states = torch.randn(2, 11, 16, requires_grad=True)
        norm, head, eps = draft.get_lm_head_params()
        output = model._calculate_loss(
            states, target, mask, 1, 11, norm, head, eps, None
        )
        gradients = torch.autograd.grad(output[0], (states, norm, head))
        return output, gradients

    torch.manual_seed(29)
    expected, expected_grad = run(True)
    torch.manual_seed(29)
    actual, actual_grad = run(False)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


def test_thd_backbone_checkpoint_is_independent_from_loss_checkpoint(monkeypatch):
    class Draft(torch.nn.Module):
        vocab_size = target_vocab_size = 4

        def get_lm_head_params(self):
            return torch.ones(4), torch.ones(4, 4), 1e-6

        def get_lm_head_up_weight(self):
            return None

        def project_hidden_states(self, states):
            return states

        def embed_input_ids(self, ids):
            return torch.zeros(*ids.shape, 4)

        def backbone(self, input_embeds, hidden_states, **_kwargs):
            return hidden_states + input_embeds, None, None

        def commit_router_load(self):
            return None

        def next_depth_hidden(self, states):
            return states

    class Layout:
        cp_size = 1
        local_length = 2
        group = None
        rope = torch.arange(2)

        def prepare_teacher_halo(self, _teacher, _length):
            return None

        def depth_inputs(self, _depth):
            return torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2)

        def teacher_at(self, teacher, _depth):
            return teacher

    checkpoint_calls = 0

    def checkpoint(fn, *args, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return fn(*args)

    monkeypatch.setattr(eagle3, "torch_checkpoint", checkpoint)
    model = eagle3.Eagle3Model(
        Draft(), length=3, gradient_checkpointing=True,
        loss_gradient_checkpointing=False,
    )
    monkeypatch.setattr(
        model,
        "_calculate_loss",
        lambda hidden_states, **_kwargs: (
            hidden_states.sum() * 0.0,
            hidden_states.new_zeros(()),
            hidden_states.new_tensor(2.0),
            hidden_states.new_zeros(()),
        ),
    )
    target = eagle3.LazyTarget(torch.zeros(1, 2, 4), torch.ones(4, 4))
    model._forward_thd(torch.zeros(1, 2, 4), target, Layout())
    assert checkpoint_calls == 3


def _trainer(cache="logits", backend="flex_attention", *, pruned=False):
    trainer = object.__new__(Eagle3Trainer)
    trainer.args = Namespace(
        eagle3_target_cache=cache,
        attention_backend=backend,
    )
    trainer.eagle3 = Namespace(length=3)
    trainer.target_lm_head_weight = torch.randn(32, 16)
    trainer.t2d = torch.ones(32, dtype=torch.bool) if pruned else None
    return trainer


def test_trainer_selects_cached_logits_without_changing_dtype():
    trainer = _trainer()
    hidden = torch.randn(2, 9, 16, dtype=torch.bfloat16)
    trainer.target_lm_head_weight = trainer.target_lm_head_weight.to(torch.bfloat16)
    target = trainer._build_padded_target(hidden, None)
    assert isinstance(target, eagle3.CachedLogitsTarget)
    assert target.target_logits_padded.shape == (2, 12, 32)
    assert target.target_logits_padded.dtype == torch.bfloat16
    assert not target.target_logits_padded.requires_grad


@pytest.mark.parametrize("backbone_checkpoint", [False, True])
@pytest.mark.parametrize("loss_checkpoint", [None, False, True])
def test_trainer_loss_checkpoint_inherits_legacy_policy_unless_overridden(
    backbone_checkpoint, loss_checkpoint
):
    trainer = _trainer()
    trainer.args.ttt_length = 3
    trainer.args.gradient_checkpointing = backbone_checkpoint
    trainer.args.eagle3_loss_gradient_checkpointing = loss_checkpoint
    model = trainer._create_training_model(_draft())
    assert model.gradient_checkpointing is backbone_checkpoint
    assert model.loss_gradient_checkpointing is (
        backbone_checkpoint if loss_checkpoint is None else loss_checkpoint
    )


@pytest.mark.parametrize(
    "trainer,match",
    [
        (_trainer(pruned=True), "full Target vocabulary"),
        (_trainer(backend="fa4"), "padded sdpa or flex_attention"),
        (_trainer(cache="unknown"), "Unsupported EAGLE3 target cache"),
    ],
)
def test_trainer_cache_contract_fails_closed(trainer, match):
    with pytest.raises(RuntimeError, match=match):
        trainer._build_padded_target(torch.randn(1, 4, 16), None)
