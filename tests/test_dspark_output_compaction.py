# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.

"""Block selection/head reuse preserve objectives, ownership and gradient hooks."""

import copy

import pytest
import torch

from aspec.models.ops.output_blocks import (
    output_block_counts,
    output_block_indices,
    owner_block_indices,
    prepare_block_layout,
    restore_blocks,
    select_blocks,
)
from tests.test_dspark_loss_alignment import objective_fixture, objective_model


@pytest.fixture(autouse=True)
def eager_l1(monkeypatch):
    # These tests exercise selection/optimizer semantics, not compiler startup.
    # Real compiled/tiled backends are checked by the separate GPU validation.
    import aspec.models.dspark as module

    monkeypatch.setattr(module, "compiled_l1_from_logits", lambda x, y:
                        (x.float().softmax(-1) - y.float().softmax(-1)).abs().sum(-1))


@pytest.mark.parametrize("keep", [
    [[True] * 6], [[True, False, True, False, False, True]], [[False] * 6],
    [[True, False, True, False], [False, True, False, False]],
])
def test_selection_is_stable_batched_and_removes_only_invalid_blocks(keep):
    mask = torch.tensor(keep)
    indices = output_block_indices(mask)
    if mask.all():
        assert indices is None
        return
    value = torch.randn(*mask.shape, 3, 5, requires_grad=True)
    selected = select_blocks(value, indices)
    selected_mask = select_blocks(mask, indices)
    for row in range(mask.shape[0]):
        torch.testing.assert_close(selected[row][selected_mask[row]], value[row][mask[row]])
    (selected * selected_mask[..., None, None]).sum().backward()
    torch.testing.assert_close(value.grad, mask[..., None, None].expand_as(value).float())


def test_counts_preserve_original_head_chunk_boundaries_and_batch_padding():
    mask = torch.tensor([[True, True, False, False, False, True, False],
                         [False, False, False, True, True, False, False]])
    assert output_block_counts(mask, 3) == [2, 2, 0]


@pytest.mark.parametrize("owners", [1, 2, 3, 4])
@pytest.mark.parametrize("empty", [False, True])
def test_combined_layout_uses_one_transfer_and_preserves_original_owners(monkeypatch, owners, empty):
    mask = torch.zeros(2, owners * 6, dtype=torch.bool)
    if not empty:
        mask[0, ::3] = True
        mask[1, -2:] = True
    expected_indices = owner_block_indices(mask, owners)
    expected_counts = [output_block_counts(mask[:, rank * 6:(rank + 1) * 6], 4)
                       for rank in range(owners)]
    original = torch.Tensor.tolist
    transfers = []

    def tolist(value):
        transfers.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("unexpected scalar transfer"))
    for rank in range(owners):
        before = len(transfers)
        indices, counts = prepare_block_layout(mask, owners, rank, 4)
        assert len(transfers) == before + 1
        assert counts == expected_counts[rank]
        if indices is None:
            assert expected_indices is None
        else:
            assert torch.equal(indices, expected_indices)


@pytest.mark.parametrize("owners", [1, 2, 3, 4])
def test_backbone_selection_preserves_owners_and_restores_valid_rows(owners):
    mask = torch.zeros(2, owners * 6, dtype=torch.bool)
    mask[0, -2:] = True
    mask[1, 1] = True
    indices = owner_block_indices(mask, owners)
    assert indices.shape == (2, owners * 2)
    expected_owners = torch.arange(owners).repeat_interleave(2).expand(2, -1)
    torch.testing.assert_close(indices // 6, expected_owners)
    value = torch.randn(2, owners * 6, 3, 5, requires_grad=True)
    restored = restore_blocks(select_blocks(value, indices), indices, owners * 6)
    torch.testing.assert_close(restored[mask], value[mask])
    (restored * mask[..., None, None]).sum().backward()
    torch.testing.assert_close(value.grad, mask[..., None, None].expand_as(value).float())


def test_empty_backbone_keeps_one_sentinel_per_owner():
    indices = owner_block_indices(torch.zeros(1, 24, dtype=torch.bool), 4)
    torch.testing.assert_close(indices, torch.tensor([[0, 6, 12, 18]]))
    assert owner_block_indices(torch.ones(1, 24, dtype=torch.bool), 4) is None


@pytest.mark.parametrize("all_masked", [False, True])
def test_backbone_restores_original_slots_before_teacher_and_loss(monkeypatch, all_masked):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                         vocab_size=23, mask_token_id=22, markov_rank=3,
                         num_target_layers=1, target_hidden_size=8,
                         target_num_hidden_layers=4, target_layer_ids=[1])
    model = DSparkModel(DSparkDraftModel(config), block_size=4, num_anchors=6,
                       attention_backend="fa4")
    batch, anchors, keep = objective_fixture(all_masked)
    batch["hidden_states_list"] = torch.randn(1, 29, 8)
    anchors = torch.where(keep, anchors, 0)
    observed = []

    def network(**kwargs):
        metadata = kwargs["direct_cp_metadata"]
        observed.append((kwargs["noise_embedding"].shape[1], metadata.anchor_positions.clone()))
        return kwargs["noise_embedding"] * 0.1

    monkeypatch.setattr(model.draft_model, "forward", network)
    hidden, returned, returned_keep, count = model._draft_backbone(
        batch["input_ids"], batch["hidden_states_list"], batch["loss_mask"],
        ctx_doc_ids=batch["ctx_doc_ids"], base_position_ids=batch["base_position_ids"],
        injected_anchors=anchors, injected_keep_mask=keep,
    )
    assert count == 6 and hidden.shape == (1, 24, 8)
    assert model.backbone_blocks_observed == (1 if all_masked else 5)
    assert observed[0][0] == (4 if all_masked else 20)
    torch.testing.assert_close(returned, anchors)
    torch.testing.assert_close(returned_keep, keep)
    expected = model._create_noise_embed(batch["input_ids"], anchors, keep) * 0.1
    torch.testing.assert_close(hidden.unflatten(1, (6, 4))[keep], expected.unflatten(1, (6, 4))[keep])
    assert observed[0][1].shape[1] == (1 if all_masked else 5)


def test_prepared_head_counts_do_not_survive_a_new_draw_or_bypassed_backbone(monkeypatch):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    config = DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                         vocab_size=23, mask_token_id=22, markov_rank=3,
                         num_target_layers=1, target_hidden_size=8,
                         target_num_hidden_layers=4, target_layer_ids=[1])
    model = DSparkModel(DSparkDraftModel(config), block_size=4, num_anchors=6, attention_backend="fa4")
    monkeypatch.setattr(model.draft_model, "forward", lambda **kwargs: kwargs["noise_embedding"] * 0.1)
    batch, anchors, keep = objective_fixture(False)
    batch["hidden_states_list"] = torch.randn(1, 29, 8)
    for current_keep in (keep, torch.zeros_like(keep), keep):
        current_anchors = torch.where(current_keep, anchors, 0)
        model(**batch, injected_anchors=current_anchors, injected_keep_mask=current_keep)
        assert model._prepared_output_counts == [int(current_keep.sum())]
        assert model.loss_head_blocks_observed == int(current_keep.sum())
    # A supported alternate/fake backbone may not prepare counts. The next
    # forward must fall back to the current mask, never retain the preceding 5.
    monkeypatch.setattr(model, "_draft_backbone", lambda *args, **kwargs: (
        torch.zeros(1, 24, 8), torch.zeros_like(anchors), torch.zeros_like(keep), 6
    ))
    model(**batch)
    assert model._prepared_output_counts is None
    assert model.loss_head_blocks_observed == 0


@pytest.mark.parametrize("teacher_trainable", [False, True])
def test_bounded_teacher_reuse_preserves_losses_and_gradients(monkeypatch, teacher_trainable):
    compact, reference, batch = models_and_batch(False)
    compact.num_anchors = reference.num_anchors = 2
    reference._compact_output_blocks = True
    reference._reuse_head_projections = False
    for model in (compact, reference):
        anchors = torch.where(batch["injected_keep_mask"], batch["injected_anchors"], 0)
        model._draft_backbone = lambda *args, _model=model, **kwargs: (
            _model.draft_model.hidden, anchors, batch["injected_keep_mask"], 6
        )
    batch["lm_head_weight"].requires_grad_(teacher_trainable)
    import aspec.models.dspark as module

    original = module.F.linear
    calls = []

    def observe(value, weight, bias=None):
        if weight is batch["lm_head_weight"]:
            calls.append(value.detach().clone())
        return original(value, weight, bias)

    monkeypatch.setattr(module.F, "linear", observe)
    outputs, gradients, projections = [], [], []
    for model in (compact, reference):
        calls.clear()
        out = model(**batch)
        outputs.append(out)
        # Include non-unit upstream scaling as in pooled accumulation. Cached
        # logits must not move scaling past a BF16 gradient computation.
        (out[0] * 0.017).backward()
        gradients.append([p.grad.clone() for p in model.parameters()])
        projections.append(len(calls))
    assert compact.loss_head_cached_teacher_blocks_observed == (0 if teacher_trainable else 2)
    assert compact.loss_head_cached_draft_blocks_observed == (0 if teacher_trainable else 2)
    assert projections[1] - projections[0] == (0 if teacher_trainable else 2)
    for actual, expected in zip(outputs[0][:5], outputs[1][:5], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def models_and_batch(all_masked):
    torch.manual_seed(73)
    compact, dense = objective_model(), objective_model()
    # Earlier stage tests retain their original one-chunk cache policy.
    compact._head_projection_cache_chunks = dense._head_projection_cache_chunks = 1
    dense.load_state_dict(compact.state_dict())
    dense._compact_output_blocks = False
    batch, anchors, keep = objective_fixture(all_masked)
    return compact, dense, {**batch, "injected_anchors": anchors, "injected_keep_mask": keep}


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("trainable_teacher", [False, True])
def test_two_chunk_projection_cache_reduces_replay_and_preserves_all_gradients(monkeypatch, context, trainable_teacher):
    actual, _, batch = models_and_batch(False)
    actual.draft_model.hidden = torch.nn.Parameter(actual.draft_model.hidden.detach().repeat(1, 4, 1))
    actual.num_anchors = 4
    actual._head_projection_cache_chunks = 2
    actual._reuse_context_teacher_projection = context
    reference = copy.deepcopy(actual)
    reference._head_projection_cache_chunks = 1
    keep = batch["injected_keep_mask"].repeat(1, 4)
    anchors = torch.where(batch["injected_keep_mask"], batch["injected_anchors"], 0).repeat(1, 4)
    for model in (actual, reference):
        model._draft_backbone = lambda *args, _model=model, **kwargs: (
            _model.draft_model.hidden, anchors, keep, 24
        )
    batch["lm_head_weight"].requires_grad_(trainable_teacher)
    import aspec.models.dspark as module

    original = module.F.linear
    projections = []

    def linear(value, weight, bias=None):
        if weight is batch["lm_head_weight"]:
            projections.append(value.shape[0])
        return original(value, weight, bias)

    monkeypatch.setattr(module.F, "linear", linear)
    outputs, gradients, calls = [], [], []
    for model in (actual, reference):
        projections.clear()
        batch["lm_head_weight"].grad = None
        out = model(**batch)
        (out[6][0] * 0.017).backward()
        outputs.append(out)
        gradients.append([p.grad.clone() for p in model.parameters()])
        if trainable_teacher:
            gradients[-1].append(batch["lm_head_weight"].grad.clone())
        calls.append(len(projections))
        retained = model.loss_head_cached_context_tokens_observed
        retained += model.block_size * (model.loss_head_cached_draft_blocks_observed + (
            0 if retained else model.loss_head_cached_teacher_blocks_observed
        ))
        assert retained <= 2 * model._head_projection_cache_chunks * max(output_block_counts(keep, 4)) * 4
    assert calls[0] < calls[1] if not trainable_teacher else calls[0] == calls[1]
    for left, right in zip(outputs[0][:5], outputs[1][:5], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("trainable_source", [None, "lm_head_weight", "last_hidden_states"])
@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("batch_rows", [1, 2])
@pytest.mark.parametrize("chunk_budget", [4, 7])
def test_context_teacher_reuse_is_bounded_and_preserves_gradients(monkeypatch, trainable_source, cp, batch_rows, chunk_budget):
    actual, reference, batch = models_and_batch(False)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key != "lm_head_weight":
            batch[key] = value.repeat(batch_rows, *([1] * (value.ndim - 1)))
    if batch_rows > 1:
        batch["last_hidden_states"][1].add_(0.3)
        batch["input_ids"][1] = (batch["input_ids"][1] + 1) % 23
    actual.draft_model.hidden = torch.nn.Parameter(actual.draft_model.hidden.detach().repeat(batch_rows, 4, 1))
    reference = copy.deepcopy(actual)
    reference._reuse_context_teacher_projection = False
    cached_inputs, projection_calls = [], []
    import aspec.models.dspark as module

    original_linear = module.F.linear
    original_terms = actual._prediction_terms

    def terms(*values):
        if values[9] is not None:
            cached_inputs.append((values[7].data_ptr(), tuple(values[7].shape)))
        return original_terms(*values)

    def linear(value, weight, bias=None):
        if weight is batch["lm_head_weight"]:
            projection_calls.append(value.shape[0])
        return original_linear(value, weight, bias)

    monkeypatch.setattr(actual, "_prediction_terms", terms)
    monkeypatch.setattr(module.F, "linear", linear)
    for model in (actual, reference):
        model.num_anchors = chunk_budget
        model.draft_context_parallel_size = cp
        keep = batch["injected_keep_mask"].repeat(1, 4)
        anchors = torch.where(batch["injected_keep_mask"], batch["injected_anchors"], 0).repeat(1, 4)
        model._draft_backbone = lambda *args, _model=model, **kwargs: (
            _model.draft_model.hidden, anchors, keep, 24
        )
        # Isolate cache policy from CP collectives; real CP backward is covered by
        # tools.validation.dspark_output_compaction, not mocked by this CPU test.
        model._align_target_hidden_states = lambda hidden, indices, **kwargs: hidden.gather(
            1, indices.flatten(1).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
    reference._compact_output_blocks = True
    if trainable_source is not None:
        batch[trainable_source].requires_grad_()
    outputs, gradients, calls = [], [], []
    for model in (actual, reference):
        projection_calls.clear()
        if trainable_source is not None:
            batch[trainable_source].grad = None
        out = model(**batch)
        (out[0] * 0.017).backward()
        outputs.append(out)
        gradients.append([p.grad.clone() for p in model.parameters()])
        if trainable_source is not None:
            gradients[-1].append(batch[trainable_source].grad.clone())
        calls.append(len(projection_calls))
    enabled = cp == 1 and trainable_source is None
    assert actual.loss_head_cached_context_tokens_observed == (batch_rows * 29 if enabled else 0)
    if enabled:
        assert calls[0] < calls[1]
        assert len(cached_inputs) > 1
        # All chunks and their backward replay see ONE shared context tensor, not
        # separately retained expanded vocabulary tensors for every query chunk.
        assert len(set(cached_inputs)) == 1
        assert cached_inputs[0][1] == (batch_rows, 29, 23)
        retained_tokens = 29 + actual.loss_head_cached_draft_blocks_observed * actual.block_size
        assert retained_tokens <= 2 * max(output_block_counts(keep, chunk_budget)) * actual.block_size
        assert actual.loss_head_cached_draft_blocks_observed == (2 if chunk_budget == 7 else 0)
    else:
        assert calls[0] == calls[1]
        assert not cached_inputs
    for out, expected in zip(outputs[0][:5], outputs[1][:5], strict=True):
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    for grad, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(grad, expected, rtol=1e-6, atol=1e-7)


def test_context_teacher_reuse_does_not_expand_the_existing_budget():
    actual, reference, batch = models_and_batch(False)
    actual.num_anchors = reference.num_anchors = 2
    keep = batch["injected_keep_mask"]
    anchors = torch.where(keep, batch["injected_anchors"], 0)
    actual._draft_backbone = lambda *args, **kwargs: (actual.draft_model.hidden, anchors, keep, 6)
    actual(**batch)
    # A 29-token row is larger than the existing two-projection 2*2*4 budget.
    assert actual.loss_head_cached_context_tokens_observed == 0
    assert actual.loss_head_cached_teacher_blocks_observed == 2
    assert actual.loss_head_cached_draft_blocks_observed == 2


def test_expanded_cache_rejects_context_projection_when_chunk_reuse_costs_less():
    actual, _, batch = models_and_batch(False)
    actual._head_projection_cache_chunks = 2
    actual.num_anchors = 2
    # Three two-block chunks: regular reuse projects 64 rows in total;
    # a 29-token context plus replay of all Draft chunks would project 77.
    keep = torch.ones(1, 6, dtype=torch.bool)
    anchors = torch.tensor([[2, 4, 13, 15, 18, 22]])
    actual._draft_backbone = lambda *args, **kwargs: (actual.draft_model.hidden, anchors, keep, 6)
    actual(**batch)
    assert actual.loss_head_cached_context_tokens_observed == 0
    assert actual.loss_head_cached_teacher_blocks_observed == 4
    assert actual.loss_head_cached_draft_blocks_observed == 4


def test_projection_budget_can_retain_more_small_original_chunks():
    from aspec.models.dspark import _cached_projection_chunks

    counts = [10, 6, 4, 3, 0]
    # Four largest projections' worth of bytes can hold three smaller pairs.
    budget = 4 * max(counts) * 7
    selected = _cached_projection_chunks(counts, budget, 7, copies=2)
    assert selected == {0, 1, 2}
    assert 2 * sum(counts[i] for i in selected) * 7 == budget
    assert _cached_projection_chunks(counts, budget, 7, copies=2, limit=2) == {0, 1}
    # One context table and four small Draft projections fit the same budget.
    selected = _cached_projection_chunks(counts, budget - 40, 7, copies=1)
    assert selected == {0, 1, 2, 3}
    assert 40 + sum(counts[i] for i in selected) * 7 <= budget
    assert _cached_projection_chunks([0, 0], budget, 7, copies=1) == set()


def test_spare_context_budget_avoids_one_projection_with_identical_head_gradients(monkeypatch):
    actual, _, batch = models_and_batch(False)
    actual.draft_model.hidden = torch.nn.Parameter(actual.draft_model.hidden.detach().repeat(1, 4, 1))
    reference = copy.deepcopy(actual)
    reference._reuse_context_draft_projection = False
    keep = batch["injected_keep_mask"].repeat(1, 4)
    anchors = torch.where(batch["injected_keep_mask"], batch["injected_anchors"], 0).repeat(1, 4)
    for model in (actual, reference):
        model.num_anchors = 7
        model._draft_backbone = lambda *args, _model=model, **kwargs: (
            _model.draft_model.hidden, anchors, keep, 24
        )
    import aspec.models.dspark as module

    original = module.F.linear
    calls = []

    def linear(value, weight, bias=None):
        if weight is batch["lm_head_weight"]:
            calls.append(value.shape[0])
        return original(value, weight, bias)

    monkeypatch.setattr(module.F, "linear", linear)
    results = []
    for model in (actual, reference):
        calls.clear()
        out = model(**batch)
        (out[0] * 0.017).backward()
        results.append((len(calls), out, [p.grad for p in model.parameters()]))
    assert results[1][0] - results[0][0] == 1
    for left, right in zip(results[0][1][:5], results[1][1][:5], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(results[0][2], results[1][2], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("all_masked", [False, True])
def test_losses_metrics_and_every_parameter_gradient_match_dense(monkeypatch, all_masked):
    compact, dense, batch = models_and_batch(all_masked)
    expected = dense(**batch)
    if all_masked:
        monkeypatch.setattr(compact, "_prediction_terms", lambda *args:
                            pytest.fail("An empty rank must not run vocabulary projections"))
    actual = compact(**batch)
    assert compact.loss_head_blocks_observed == (0 if all_masked else 5)
    for index in (0, 1, 2, 3, 4):
        torch.testing.assert_close(actual[index], expected[index], atol=1e-6, rtol=1e-6)
    for key in actual[5]:
        torch.testing.assert_close(actual[5][key], expected[5][key])
    torch.testing.assert_close(actual[6], expected[6])
    actual[0].backward()
    expected[0].backward()
    for (name, parameter), (other_name, reference) in zip(
        compact.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert name == other_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-6, rtol=1e-6)


def test_sixteen_optimizer_updates_and_resume_match_dense():
    compact, dense, batch = models_and_batch(False)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=6e-4) for model in (compact, dense)]
    for step in range(16):
        for model, optimizer in zip((compact, dense), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            # Pool two additive microbatches, including an empty accumulation tail.
            first = model(**batch)[6]
            empty_batch = {**batch, "injected_keep_mask": torch.zeros_like(batch["injected_keep_mask"])}
            second = model(**empty_batch)[6]
            ((first[0] + second[0]) / (first[1] + second[1]).clamp(min=1e-6)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for parameter, reference in zip(compact.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(parameter, reference, atol=1e-6, rtol=1e-6)
        if step == 7:
            # Reconstruct the wrapper as well as Adam's moments, not just reload weights.
            state, optimizer_state = copy.deepcopy(compact.state_dict()), copy.deepcopy(optimizers[0].state_dict())
            compact = objective_model()
            compact.load_state_dict(state)
            optimizers[0] = torch.optim.AdamW(compact.parameters(), lr=6e-4)
            optimizers[0].load_state_dict(optimizer_state)


def test_empty_rank_does_not_hide_nonfinite_backbone():
    compact, _, batch = models_and_batch(True)
    with torch.no_grad():
        compact.draft_model.hidden[0, 0, 0] = float("nan")
    assert not torch.isfinite(compact(**batch)[0])


def test_auxiliary_objective_retains_full_backbone_query_shape():
    compact, _, batch = models_and_batch(False)
    observed = []

    def auxiliary(router_logits, weights, denominator):
        observed.append(weights.shape)
        return weights.new_zeros(()), {}

    compact._moe_training_loss = auxiliary
    compact(**batch)
    assert observed == [(1, 6, 4)]
    assert compact.loss_head_blocks_observed == 5


def test_compaction_does_not_merge_original_gradient_chunks(monkeypatch):
    compact, _, batch = models_and_batch(False)
    compact.num_anchors = 2
    keep = torch.tensor([[True, False, False, True, True, False]])
    anchors = torch.where(keep, batch["injected_anchors"], 0)
    compact._draft_backbone = lambda *args, **kwargs: (compact.draft_model.hidden, anchors, keep, 6)
    calls = []
    original = compact._prediction_terms

    def observe(hidden, targets, previous, *args):
        calls.append(previous[:, :, 0].detach().clone())
        return original(hidden, targets, previous, *args)

    monkeypatch.setattr(compact, "_prediction_terms", observe)
    compact(**batch)
    assert compact.loss_head_chunks_observed == 3
    assert [value.shape[1] for value in calls] == [1, 1, 1]
    expected = batch["input_ids"][:, [2, 18, 22]]
    torch.testing.assert_close(torch.cat(calls, 1), expected)


@pytest.mark.parametrize("all_masked", [False, True])
def test_disabled_confidence_preserves_none_gradients(all_masked):
    compact, dense, batch = models_and_batch(all_masked)
    for model in (compact, dense):
        model.confidence_head_alpha = 0
        model(**batch)[0].backward()
    for (name, actual), (_, reference) in zip(compact.named_parameters(), dense.named_parameters(), strict=True):
        if "confidence_head" in name:
            assert actual.grad is None and reference.grad is None
        else:
            torch.testing.assert_close(actual.grad, reference.grad, atol=1e-6, rtol=1e-6)


def test_disabled_confidence_does_not_update_existing_adam_moments_or_decay():
    model, _, batch = models_and_batch(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    model(**batch)[0].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    head = list(model.draft_model.confidence_head.parameters())
    weights = [value.detach().clone() for value in head]
    moments = [copy.deepcopy(optimizer.state[value]) for value in head]
    model.confidence_head_alpha = 0
    batch["injected_keep_mask"].zero_()
    model(**batch)[0].backward()
    assert all(value.grad is None for value in head)
    optimizer.step()
    for value, before, state in zip(head, weights, moments, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
        for field in state:
            torch.testing.assert_close(optimizer.state[value][field], state[field], atol=0, rtol=0)


@pytest.mark.parametrize("batch_rows", [1, 3])
@pytest.mark.parametrize("all_masked", [False, True])
def test_native_bshd_compacts_backbone_and_restores_original_slots(monkeypatch, batch_rows, all_masked):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    config = DSparkConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                         vocab_size=23, mask_token_id=22, markov_rank=3,
                         num_target_layers=1, target_hidden_size=8,
                         target_num_hidden_layers=4, target_layer_ids=[1])
    model = DSparkModel(DSparkDraftModel(config), block_size=4, num_anchors=6,
                       attention_backend="cudnn_flex_attention")
    batch, anchors, keep = objective_fixture(all_masked)
    anchors, keep = anchors.repeat(batch_rows, 1), keep.repeat(batch_rows, 1)
    if batch_rows > 1:
        keep[-1].zero_()
    anchors = torch.where(keep, anchors, 0)
    ids = batch["input_ids"].repeat(batch_rows, 1)
    masks = batch["loss_mask"].repeat(batch_rows, 1)
    observed = []

    def network(**kwargs):
        assert kwargs["direct_cp_metadata"] is None
        observed.append(kwargs["noise_embedding"].shape[1])
        return kwargs["noise_embedding"] * 0.1

    monkeypatch.setattr(model.draft_model, "forward", network)
    hidden, returned, returned_keep, count = model._draft_backbone(
        ids, torch.randn(batch_rows, 29, 8), masks,
        injected_anchors=anchors, injected_keep_mask=keep,
    )
    assert count == 6 and hidden.shape == (batch_rows, 24, 256)
    assert observed == [4 if all_masked else 20]
    torch.testing.assert_close(returned, anchors)
    torch.testing.assert_close(returned_keep, keep)
    expected = model._create_noise_embed(ids, anchors, keep) * 0.1
    torch.testing.assert_close(hidden.unflatten(1, (6, 4))[keep], expected.unflatten(1, (6, 4))[keep])
    assert model._prepared_output_counts == [0 if all_masked else 5]


@pytest.mark.parametrize("batch_rows", [1, 2, 8, 16])
@pytest.mark.parametrize("trainable_source", [None, "lm_head_weight", "last_hidden_states"])
def test_single_chunk_context_reuse_has_a_batch_independent_bound(monkeypatch, batch_rows, trainable_source):
    actual, reference, batch = models_and_batch(False)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key != "lm_head_weight":
            batch[key] = value.repeat(batch_rows, *([1] * (value.ndim - 1)))
    anchors = torch.where(batch["injected_keep_mask"], batch["injected_anchors"], 0).repeat(1, 3)
    keep = batch["injected_keep_mask"].repeat(1, 3)
    batch["injected_anchors"], batch["injected_keep_mask"] = anchors, keep
    if batch_rows > 1:
        batch["last_hidden_states"][1].add_(0.3)
    if trainable_source:
        batch[trainable_source].requires_grad_()
    for current in (actual, reference):
        current.num_anchors = 18
        current._head_projection_cache_chunks = 2
        current._compact_output_blocks = True
        current.draft_model.hidden = torch.nn.Parameter(current.draft_model.hidden.detach().repeat(batch_rows, 3, 1))
        current._draft_backbone = lambda *a, _current=current, **kw: (_current.draft_model.hidden, anchors, keep, 18)
    reference._reuse_head_projections = False
    reference._reuse_context_teacher_projection = False
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda *a, **kw:
                        pytest.fail("A single head chunk must not start checkpoint replay"))
    outputs, gradients = [], []
    for current in (actual, reference):
        if trainable_source:
            batch[trainable_source].grad = None
        out = current(**batch)
        (out[0] * 0.017).backward()
        outputs.append(out[0].detach())
        gradients.append([p.grad.clone() for p in current.parameters()])
        if trainable_source:
            gradients[-1].append(batch[trainable_source].grad.clone())
    enabled = trainable_source is None and batch_rows * 29 <= 240
    assert actual.loss_head_cached_context_tokens_observed == (batch_rows * 29 if enabled else 0)
    assert actual.loss_head_cached_draft_blocks_observed == 0
    assert actual.loss_head_chunks_observed == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-6, atol=1e-6)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
