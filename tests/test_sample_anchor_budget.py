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

"""Per-source-sample anchor semantics without GPU compilation."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aspec.config.train_config import load_config
from aspec.data.utils import DFlashPackingCollator
from aspec.models.dflash import DFlashModel
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.dspark import DSparkModel
from aspec.models.ops.anchors import (
    document_anchor_slots,
    sample_anchor_positions,
    sample_document_anchor_positions,
)
from aspec.planning import compile_execution_plan
from aspec.training.checkpoint import _logical_training_topology, _validate_exact_resume_topology


@pytest.mark.parametrize("anchors", [4, 512])
@pytest.mark.parametrize("lengths", [(9, 13, 30), (80,), (20, 51)])
def test_segmented_sampling_matches_native_per_document(lengths, anchors):
    masks = [torch.ones(1, n) for n in lengths]
    for mask in masks:
        mask[:, 2:5] = 0
    docs = torch.cat([torch.full((1, n), i) for i, n in enumerate(lengths)], 1)
    generator = torch.Generator().manual_seed(91)
    actual, keep = sample_document_anchor_positions(
        torch.cat(masks, 1), docs, lengths, anchors, 4, generator
    )
    generator.manual_seed(91)
    slot_counts, total, padded = document_anchor_slots(lengths, anchors, 4)
    offset = cursor = 0
    for mask, length, slots in zip(masks, lengths, slot_counts, strict=True):
        expected, valid = sample_anchor_positions(length, mask, anchors, 4, mask.device,
                                                 generator=generator)
        torch.testing.assert_close(keep[:, cursor:cursor + slots], valid[:, :slots])
        torch.testing.assert_close(actual[:, cursor:cursor + slots],
                                   torch.where(valid[:, :slots], expected[:, :slots] + offset, 0))
        offset += length
        cursor += slots
    assert actual.numel() == padded == total


def test_empty_short_and_unsupervised_documents_do_not_borrow_budget():
    lengths = (3, 19, 4, 31)
    mask = torch.ones(1, sum(lengths))
    mask[:, 3:22] = 0
    docs = torch.cat([torch.full((1, n), i) for i, n in enumerate(lengths)], 1)
    anchors, keep = sample_document_anchor_positions(mask, docs, lengths, 8, 4, alignment=4)
    assert anchors.numel() % 4 == 0
    assert keep.sum() == 8
    assert (docs.gather(1, anchors)[keep] == 3).all()
    empty = sample_document_anchor_positions(mask.zero_(), docs, lengths, 8, 4)
    assert not empty[1].any()
    assert not empty[0].any()
    short = sample_document_anchor_positions(torch.ones(1, 7), docs[:, :7], (3, 4), 512, 4)
    assert short[0].shape == (1, 1) and not short[1].any()


def test_short_documents_do_not_allocate_512_useless_query_slots_each():
    slots, total, padded = document_anchor_slots((33,) * 100, 512, 7, 4)
    assert slots == (27,) * 100
    assert total == padded == 2700


def test_compacted_budget_masks_only_the_extra_cp_alignment_slots():
    mask = torch.ones(1, 22)
    docs = torch.tensor([[0] * 9 + [1] * 13])
    anchors, keep = sample_document_anchor_positions(mask, docs, (9, 13), 8, 7, alignment=4)
    assert anchors.shape == keep.shape == (1, 12)
    assert keep.sum() == 10
    assert not keep[0, -2:].any() and not anchors[0, -2:].any()
    assert (docs.gather(1, anchors)[0, :3] == 0).all()
    assert (docs.gather(1, anchors)[0, 3:10] == 1).all()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_noncontiguous_head_chunks_use_token_major_matrix_projections(monkeypatch, batch_size):
    import aspec.models.dspark as module

    draft = torch.nn.Module()
    draft.markov_head = draft.confidence_head = None
    model = DSparkModel(draft, block_size=4, num_anchors=4, confidence_head_alpha=0)
    base = torch.randn(batch_size, 8, 4, 8, requires_grad=True)
    hidden = base[:, 1:5]
    # At B=1, is_contiguous is True despite the parent row's retained leading
    # stride. This also triggered tiny broadcast GEMMs in real packed training.
    assert hidden.is_contiguous() == (batch_size == 1)
    target = torch.randn_like(base)[:, 1:5]
    ids = torch.randint(0, 23, hidden.shape[:-1])
    weight = torch.nn.Parameter(torch.randn(23, 8), requires_grad=False)
    mask = torch.ones_like(ids, dtype=torch.float32)
    dimensions = []
    original = module.F.linear

    def observed(value, matrix, *args, **kwargs):
        if matrix is weight:
            dimensions.append(value.ndim)
        return original(value, matrix, *args, **kwargs)

    monkeypatch.setattr(module.F, "linear", observed)
    # This test checks projection layout; compiled/backend numerical checks are separate.
    monkeypatch.setattr(module, "compiled_l1_from_logits",
                        lambda x, y: (x.float().softmax(-1) - y.float().softmax(-1)).abs().sum(-1))
    terms = model._prediction_terms(hidden, ids, ids, weight, target, mask, mask)
    assert dimensions == [2, 2]
    (terms[0] + terms[1]).backward()
    assert torch.isfinite(base.grad).all()


def test_sample_scope_requires_collator_metadata_and_validates_replay_budget():
    model = DFlashModel(torch.nn.Module(), block_size=4, num_anchors=4,
                        anchor_sampling_scope="sample")
    mask = torch.ones(1, 32)
    docs = torch.tensor([[0] * 16 + [1] * 16])
    with pytest.raises(ValueError, match="packed_document_lengths"):
        model._sample_anchor_positions(32, mask, mask.device, ctx_doc_ids=docs)
    with pytest.raises(ValueError, match="shape"):
        model._sample_anchor_positions(32, mask, mask.device, ctx_doc_ids=docs,
                                       packed_document_lengths=(16, 16),
                                       injected_anchors=torch.zeros(1, 4, dtype=torch.long),
                                       injected_keep_mask=torch.zeros(1, 4, dtype=torch.bool))


def test_sample_policy_is_in_plan_and_resume_identity():
    root = Path(__file__).parents[1]
    config = load_config(str(root / "configs/examples/sglang_qwen3_8b_dspark.yaml"),
                         ["training.anchor_sampling_scope=sample"])
    plan = compile_execution_plan(config).to_dict()
    assert plan["workload"]["objective"]["anchor_sampling_scope"] == "sample"
    args = SimpleNamespace(draft_algorithm="dspark", dflash_block_size=7,
                           anchor_sampling_scope="sample", world_size=1)
    assert _logical_training_topology(args, 1)["training_objective"]["anchor_sampling_scope"] == "sample"
    with pytest.raises(RuntimeError, match="anchor_sampling_scope"):
        _validate_exact_resume_topology(args, {"world_size": 1})
    args.anchor_sampling_scope = "row"
    _validate_exact_resume_topology(args, {"world_size": 1})


def test_packed_dspark_loss_and_gradient_equal_independent_document_terms():
    torch.manual_seed(71)
    config = DSparkConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=67,
                         num_target_layers=1, target_hidden_size=32,
                         target_num_hidden_layers=4, target_layer_ids=[1],
                         mask_token_id=66, markov_rank=8)
    packed_model = DSparkModel(DSparkDraftModel(config), block_size=4, num_anchors=4,
                              anchor_sampling_scope="sample")
    # CPU's ordinary Draft fallback is unmasked bidirectional SDPA, not the
    # packed training path. Use the existing dense Direct-CP math oracle instead.
    packed_model.direct_cp_backend = True
    packed_model.attention_backend = "math"
    independent = copy.deepcopy(packed_model)
    independent.anchor_sampling_scope = "row"
    items = []
    for i, length in enumerate((13, 19)):
        items.append({"input_ids": torch.randint(0, 67, (1, length)),
                      "loss_mask": torch.ones(1, length),
                      "hidden_states": torch.randn(1, length, 32),
                      "last_hidden_states": torch.randn(1, length, 32),
                      "_aspec_work": {"work_id": str(i), "execution_id": str(i)}})
    batch = DFlashPackingCollator(32, pad_to_max_length=False)(items)
    anchors = torch.tensor([[2, 4, 7, 9, 14, 17, 22, 28]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    head = torch.randn(67, 32)
    result = packed_model(input_ids=batch["input_ids"], hidden_states_list=batch["hidden_states"],
                          loss_mask=batch["loss_mask"], lm_head_weight=head,
                          last_hidden_states=batch["last_hidden_states"],
                          ctx_doc_ids=batch["ctx_doc_ids"], base_position_ids=batch["base_position_ids"],
                          packed_document_lengths=batch["packed_document_lengths"],
                          injected_anchors=anchors, injected_keep_mask=keep)
    assert packed_model.loss_head_chunks_observed == 2
    terms = []
    offset = 0
    for i, item in enumerate(items):
        out = independent(input_ids=item["input_ids"], hidden_states_list=item["hidden_states"],
                          loss_mask=item["loss_mask"], lm_head_weight=head,
                          last_hidden_states=item["last_hidden_states"],
                          ctx_doc_ids=torch.zeros_like(item["input_ids"]),
                          injected_anchors=anchors[:, i * 4:(i + 1) * 4] - offset,
                          injected_keep_mask=keep[:, i * 4:(i + 1) * 4])
        terms.append(out[6])
        offset += item["input_ids"].shape[1]
    numerator = sum(term[0] for term in terms)
    denominator = sum(term[1] for term in terms)
    torch.testing.assert_close(result[6][1], denominator)
    torch.testing.assert_close(result[6][0], numerator, rtol=1e-5, atol=1e-5)
    result[0].backward()
    (numerator / denominator).backward()
    for (name, p), (_, q) in zip(packed_model.named_parameters(), independent.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)
