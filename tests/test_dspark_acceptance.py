# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from aspec.models.ops.acceptance import acceptance_components, acceptance_statistics
from tools.validation.dspark_nonpacking_alignment import (
    ReferenceAcceptanceObserver,
    training_acceptance,
)


def test_probability_tau_averages_block_prefixes_not_marginal_products():
    alpha = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    statistics = acceptance_statistics(alpha, torch.ones_like(alpha))
    torch.testing.assert_close(statistics, torch.tensor([[1.0, 1.0, 3.0], [2.0, 2.0, 2.0]]))
    assert not statistics.requires_grad
    rates = statistics[0] / statistics[1]
    assert rates[-1] == 1.5
    assert 1 + rates[:-1].cumprod(0).sum() == 1.75


def test_tail_mask_and_empty_blocks_do_not_invent_target_bonus_tokens():
    alpha = torch.tensor([[[0.5, 0.8, 0.9], [0.2, 0.9, 1.0], [1.0, 1.0, 1.0]]])
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
    result = acceptance_statistics(alpha, mask)
    torch.testing.assert_close(result, torch.tensor([[0.7, 0.8, 0.0, 3.1], [2.0, 1.0, 0.0, 2.0]]))
    parts = [acceptance_statistics(alpha[:, i:i + 1], mask[:, i:i + 1]) for i in range(3)]
    torch.testing.assert_close(sum(parts), result)
    assert acceptance_statistics(alpha, torch.zeros_like(mask)).count_nonzero() == 0


def test_probability_acceptance_keeps_position_and_block_denominators():
    statistics = acceptance_statistics(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), torch.ones(1, 2, 2))
    positions = torch.tensor([[2.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
    result = training_acceptance(positions, {"l1": 1.0}, statistics)
    assert result["per_position_accept_rate"] == [0.5, 0.5]
    assert result["tau_probabilistic"] == 1.5
    assert result["expected_accepted_draft_tokens"] == 0.5
    assert result["acceptance_block_count"] == 2
    assert result["simulated_accepted_draft_tokens"] == 0.75
    assert set(acceptance_components(statistics)) == {"accept_rate@0", "accept_rate@1", "tau_probabilistic"}
    with pytest.raises(ValueError, match="supervision"):
        training_acceptance(positions, {"l1": 1.0}, statistics * 2)


def test_reference_observer_preserves_native_bce_output_gradients_and_rng():
    observer = ReferenceAcceptanceObserver(torch.nn.functional)
    prediction = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.rand_like(prediction)
    observer.mask = torch.tensor([[[1, 1, 0], [1, 0, 0]]])
    rng = torch.random.get_rng_state().clone()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    actual = observer.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(observer.statistics, acceptance_statistics(target, observer.mask))
    assert observer.linear is torch.nn.functional.linear


def test_model_acceptance_observation_does_not_change_head_loss_or_gradients(monkeypatch):
    import aspec.models.dspark as module

    class ConfidenceHead(torch.nn.Linear):
        def forward(self, value):
            return super().forward(value).squeeze(-1)

    draft = torch.nn.Module()
    draft.markov_head = None
    draft.confidence_head = ConfidenceHead(4, 1)
    draft.confidence_head_with_markov = False
    model = module.DSparkModel(draft, block_size=2, num_anchors=2)
    monkeypatch.setattr(module, "compiled_l1_from_logits", lambda x, y:
                        (x.float().softmax(-1) - y.float().softmax(-1)).abs().sum(-1))
    hidden = torch.randn(1, 2, 2, 4, requires_grad=True)
    teacher = torch.randn_like(hidden)
    head = torch.randn(11, 4)
    labels = torch.randint(0, 11, (1, 2, 2))
    mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])

    def run():
        output = model._prediction_terms(hidden, labels, labels, head, teacher, mask, mask)
        loss = 0.1 * output[0] + 0.9 * output[1] + output[2]
        gradients = torch.autograd.grad(loss, (hidden, *draft.confidence_head.parameters()))
        return output, gradients

    observed, gradients = run()
    assert observed[-1].shape == (2, 3) and not observed[-1].requires_grad
    monkeypatch.setattr(module, "acceptance_statistics", lambda a, m: a.new_zeros((2, 3)))
    plain, plain_gradients = run()
    for actual, expected in zip(observed[:5], plain[:5], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(gradients, plain_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
