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

"""Fast CPU objective/sampling regressions; no GPU kernel compilation."""

import copy
from types import SimpleNamespace

import pytest
import torch

from aspec.config.train_config import TrainingConfig
from aspec.models.dflash import DFlashModel
from aspec.models.draft.dspark import AcceptRatePredictor, VanillaMarkov
from aspec.models.dspark import DSparkModel
from aspec.models.ops.anchors import anchor_sampling_seed, sample_anchor_positions
from aspec.training.dflash_trainer import DFlashTrainer
from aspec.training.dspark_trainer import DSparkTrainer
from tools.validation.dspark_reference import official_dspark_terms, official_loss


class TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Parameter(torch.randn(1, 24, 8) * 0.1)
        self.markov_head = VanillaMarkov(vocab_size=23, markov_rank=3)
        self.confidence_head = AcceptRatePredictor(11)
        self.confidence_head_with_markov = True


def objective_fixture(all_masked=False):
    generator = torch.Generator().manual_seed(91)
    batch = {
        "input_ids": torch.randint(0, 23, (1, 29), generator=generator),
        "hidden_states_list": [],
        "loss_mask": torch.ones(1, 29),
        "lm_head_weight": torch.randn(23, 8, generator=generator) * 0.1,
        "last_hidden_states": torch.randn(1, 29, 8, generator=generator),
        "ctx_doc_ids": torch.tensor([[0] * 12 + [1] * 17]),
        "base_position_ids": torch.tensor([list(range(12)) + list(range(17))]),
    }
    batch["loss_mask"][:, [0, 9, 12, 20]] = 0
    anchors = torch.tensor([[2, 7, 13, 18, 22, 999]])
    keep = torch.tensor([[True, True, True, True, True, False]])
    if all_masked:
        batch["loss_mask"].zero_()
        keep.zero_()
    return batch, anchors, keep


def objective_model():
    draft = TinyDraft()
    model = DSparkModel(draft, block_size=4, num_anchors=6)

    def backbone(input_ids, hidden_states_list, loss_mask, **kwargs):
        anchors, keep = model._sample_anchor_positions(
            input_ids.shape[1],
            loss_mask,
            input_ids.device,
            ctx_doc_ids=kwargs.get("ctx_doc_ids"),
            injected_anchors=kwargs.get("injected_anchors"),
            injected_keep_mask=kwargs.get("injected_keep_mask"),
        )
        return draft.hidden, anchors, keep, anchors.shape[1]

    model._draft_backbone = backbone
    return model


@pytest.mark.parametrize("all_masked", [False, True])
def test_loss_components_labels_and_gradients_match_official_deepspec(all_masked):
    torch.manual_seed(73)
    model = objective_model()
    reference = copy.deepcopy(model.draft_model)
    batch, anchors, keep = objective_fixture(all_masked)
    actual = model(**batch, injected_anchors=anchors, injected_keep_mask=keep)
    safe_anchors = torch.where(keep, anchors, 0)
    terms = official_dspark_terms(reference.hidden, reference, batch, safe_anchors, keep, 4)
    expected = official_loss(terms)
    torch.testing.assert_close(actual[0], expected, atol=1e-6, rtol=1e-6)
    for name in ("ce_loss", "l1_loss", "confidence_loss"):
        for value, ref in zip(actual[5][name], terms[name], strict=True):
            torch.testing.assert_close(value, ref)
    torch.testing.assert_close(actual[4], terms["eval_mask"].float().sum((0, 1)))
    if not all_masked:
        assert terms["target_ids"][0, 0].tolist() == batch["input_ids"][0, 3:7].tolist()
        assert terms["teacher_indices"][0, 0].tolist() == [2, 3, 4, 5]
        assert terms["previous_ids"][0, 0].tolist() == batch["input_ids"][0, 2:6].tolist()
        assert terms["eval_mask"][0, 1].tolist() == [True, False, False, False]
        assert actual[4][0] == 5  # slot zero is productive, not an anchor-only slot
    actual[0].backward()
    expected.backward()
    for (name, p), (ref_name, q) in zip(
        model.draft_model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-6)


def test_legacy_dflash_label_and_slot_zero_policy_is_detected():
    model = objective_model()
    batch, anchors, keep = objective_fixture()
    anchors = torch.where(keep, anchors, 0)
    terms = official_dspark_terms(
        model.draft_model.hidden, model.draft_model, batch, anchors, keep, 4
    )
    legacy_labels = anchors[..., None] + torch.arange(4)
    legacy_mask = terms["eval_mask"].clone()
    legacy_mask[..., 0] = False
    assert not torch.equal(legacy_labels, anchors[..., None] + torch.arange(1, 5))
    assert legacy_mask.sum() < terms["eval_mask"].sum()


@pytest.mark.parametrize("trainer_class", [DFlashTrainer, DSparkTrainer])
def test_trainer_propagates_default_off_and_configured_anchor_rng(trainer_class):
    trainer = trainer_class.__new__(trainer_class)
    trainer.args = SimpleNamespace(seed=13, deterministic_anchor_sampling=True,
                                   anchor_sampling_scope="sample")
    trainer.block_size, trainer.num_anchors = 4, 6
    trainer.loss_objective, trainer.dpace_alpha = "decay", 0.5
    trainer.loss_decay_gamma = 4.0
    trainer.ce_loss_alpha, trainer.l1_loss_alpha = 0.1, 0.9
    trainer.confidence_head_alpha = 1.0
    wrapper = trainer._build_training_wrapper(TinyDraft())
    assert wrapper.deterministic_anchor_sampling is True
    assert wrapper.anchor_seed == 13
    assert wrapper.anchor_sampling_scope == "sample"
    assert TrainingConfig().deterministic_anchor_sampling is False
    assert TrainingConfig().anchor_sampling_scope == "row"


def test_default_random_sampling_preserves_rng_behavior():
    model = DFlashModel(TinyDraft(), block_size=4, num_anchors=6)
    mask = torch.ones(2, 80)
    torch.manual_seed(55)
    actual = model._sample_anchor_positions(80, mask, mask.device)
    state = torch.get_rng_state()
    torch.manual_seed(55)
    expected = sample_anchor_positions(80, mask, 6, 4, mask.device)
    assert torch.equal(state, torch.get_rng_state())
    for value, ref in zip(actual, expected, strict=True):
        assert torch.equal(value, ref)


def test_deterministic_sampling_is_stateless_and_does_not_touch_global_rng():
    model = DFlashModel(
        TinyDraft(), block_size=4, num_anchors=6, deterministic_anchor_sampling=True, anchor_seed=42
    )
    mask = torch.ones(2, 80)
    state = torch.get_rng_state()
    first = model._sample_anchor_positions(80, mask, mask.device)
    assert torch.equal(state, torch.get_rng_state())
    torch.rand(79)
    second = model._sample_anchor_positions(80, mask, mask.device)
    for value, ref in zip(first, second, strict=True):
        assert torch.equal(value, ref)
    keys = [
        anchor_sampling_seed(42, step, mb, replica)
        for step in range(3)
        for mb in range(2)
        for replica in range(2)
    ]
    assert len(set(keys)) == len(keys)
    third = model._sample_anchor_positions(80, mask, mask.device, anchor_seed=keys[1])
    assert not torch.equal(first[0], third[0])


@pytest.mark.parametrize(
    "problem", ["missing_keep", "shape", "dtype", "bounds", "supervision", "boundary"]
)
def test_bad_anchor_replay_fails_closed(problem):
    model = objective_model()
    batch, anchors, keep = objective_fixture()
    if problem == "missing_keep":
        keep = None
    elif problem == "shape":
        anchors = anchors[:, :2]
    elif problem == "dtype":
        anchors = anchors.float()
    elif problem == "bounds":
        anchors[0, 0] = 29
    elif problem == "supervision":
        anchors[0, 0] = 8  # first target has mask zero
    elif problem == "boundary":
        anchors[0, 0] = 10
    with pytest.raises(ValueError):
        model(**batch, injected_anchors=anchors, injected_keep_mask=keep)


def test_anchor_mode_is_part_of_canonical_workload():
    from pathlib import Path

    from aspec.config import load_config
    from aspec.planning import compile_execution_plan

    path = Path(__file__).resolve().parents[1] / "configs/examples/sglang_qwen3_8b_dspark.yaml"
    original = compile_execution_plan(load_config(str(path)))
    enabled = compile_execution_plan(
        load_config(str(path), cli_args=["training.deterministic_anchor_sampling=true"])
    )
    assert original.workload.objective["deterministic_anchor_sampling"] is False
    assert enabled.workload.objective["deterministic_anchor_sampling"] is True
    assert original.plan_id != enabled.plan_id


def test_training_step_passes_logical_seed_without_cuda_or_new_rng_state(monkeypatch):
    trainer = DFlashTrainer.__new__(DFlashTrainer)
    trainer.args = SimpleNamespace(seed=19, deterministic_anchor_sampling=True)
    trainer.global_step, trainer.dp_rank = 7, 2
    observed = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)

    def forward(batch, *, anchor_seed):
        observed.append(anchor_seed)
        scalar = torch.tensor(1.0)
        return scalar, scalar, scalar[None], scalar[None], scalar[None], {}, (scalar, scalar)

    trainer._forward = forward
    trainer._backward = lambda *args, **kwargs: None
    trainer._normalize_accumulated_gradients = lambda *args: None
    trainer._train_step({}, 1, 999, 0, 1)
    trainer._train_step({}, 1, 999, 0, 1)  # exact resume's same saved optimizer step
    assert observed == [anchor_sampling_seed(19, 7, 0, 2)] * 2
