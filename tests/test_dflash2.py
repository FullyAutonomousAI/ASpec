# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest

import torch
import torch.nn.functional as F

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.dflash2 import DFlash2Model
from aspec.models.draft.dflash2 import (
    CandidateSelector,
    DFlash2Config,
    DFlash2DraftModel,
    DFlashGroupedConv,
)


def _tiny_config_kwargs(**overrides):
    config = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "num_target_layers": 4,
        "block_size": 2,
        "conv_group_size": 2,
        "selector_rank": 2,
        "selector_top_k": 2,
        "mask_token_id": 7,
    }
    config.update(overrides)
    return config


def _make_config(
    hidden_size=16,
    vocab_size=32,
    num_target_layers=2,
    block_size=4,
    selector_rank=4,
    selector_top_k=None,
):
    num_attention_heads = min(4, hidden_size // 2)
    return DFlash2Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000.0,
        num_target_layers=num_target_layers,
        target_hidden_size=hidden_size,
        target_num_hidden_layers=12,
        target_layer_ids=[1, 9][:num_target_layers],
        mask_token_id=vocab_size - 1,
        block_size=block_size,
        conv_kernel_size=2,
        conv_group_size=min(4, hidden_size),
        selector_rank=selector_rank,
        selector_top_k=selector_top_k if selector_top_k is not None else min(4, vocab_size),
    )


def _make_model(selector_loss_alpha=0.4):
    config = _make_config()
    draft = DFlash2DraftModel(config).to(dtype=torch.float32)
    draft.freeze_embedding()
    return DFlash2Model(
        draft_model=draft,
        block_size=config.block_size,
        num_anchors=2,
        loss_decay_gamma=4.0,
        selector_loss_alpha=selector_loss_alpha,
    )


def _batch(seed=0, all_masked=False):
    generator = torch.Generator().manual_seed(seed)
    batch_size, sequence_length, hidden_size, vocab_size = 2, 12, 16, 32
    loss_mask = torch.zeros(batch_size, sequence_length)
    if not all_masked:
        loss_mask[:, 2:] = 1
    return {
        "input_ids": torch.randint(
            0, vocab_size - 1, (batch_size, sequence_length), generator=generator
        ),
        "hidden_states_list": [
            torch.randn(batch_size, sequence_length, hidden_size, generator=generator)
            for _ in range(2)
        ],
        "loss_mask": loss_mask,
        "lm_head_weight": torch.randn(vocab_size, hidden_size, generator=generator),
    }


class TestDFlashGroupedConv(unittest.TestCase):
    def setUp(self):
        self.conv = DFlashGroupedConv(
            hidden_size=4,
            kernel_size=2,
            group_size=2,
            block_size=3,
        )
        with torch.no_grad():
            self.conv.kernel_projection.weight.zero_()
            self.conv.base_kernel.zero_()
            self.conv.base_kernel[0, 1].fill_(1)
            self.conv.base_kernel[1, 0].fill_(2)
            self.conv.base_kernel[1, 1].fill_(3)

    def test_parameter_schema_and_block_boundary(self):
        self.assertEqual(self.conv.base_kernel.shape, (2, 2, 4))
        self.assertEqual(self.conv.kernel_projection.weight.shape, (8, 4))
        hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
        with torch.no_grad():
            self.conv.kernel_projection.weight[:, 0] = torch.tensor(
                [0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6]
            )

        prepared, finish_kernel = self.conv.prepare(hidden)

        shifted = torch.zeros_like(hidden)
        shifted[:, 1:3] = hidden[:, 0:2]
        shifted[:, 4:6] = hidden[:, 3:5]
        position_scale = hidden[..., :1]
        expected = 0.1 * position_scale * hidden + (1 + 0.2 * position_scale) * shifted
        torch.testing.assert_close(prepared, expected)
        self.assertEqual(finish_kernel.shape, (1, 6, 2, 2))
        torch.testing.assert_close(finish_kernel[..., 0, 0], 0.3 * position_scale[..., 0])
        torch.testing.assert_close(finish_kernel[..., 0, 1], 0.4 * position_scale[..., 0])
        torch.testing.assert_close(finish_kernel[..., 1, 0], 0.5 * position_scale[..., 0])
        torch.testing.assert_close(finish_kernel[..., 1, 1], 0.6 * position_scale[..., 0])

        finished = self.conv.finish(prepared, finish_kernel)
        shifted = torch.zeros_like(prepared)
        shifted[:, 1:3] = prepared[:, 0:2]
        shifted[:, 4:6] = prepared[:, 3:5]
        expected = torch.empty_like(finished)
        expected[..., :2] = (2 + 0.3 * position_scale) * prepared[..., :2] + (
            3 + 0.5 * position_scale
        ) * shifted[..., :2]
        expected[..., 2:] = (2 + 0.4 * position_scale) * prepared[..., 2:] + (
            3 + 0.6 * position_scale
        ) * shifted[..., 2:]
        torch.testing.assert_close(finished, expected)

    def test_output_is_causal_within_each_block(self):
        with torch.no_grad():
            self.conv.kernel_projection.weight.copy_(
                torch.arange(32, dtype=torch.float32).view(8, 4) / 100
            )
        hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
        changed = hidden.clone()
        changed[:, 2] = -1000

        baseline, _ = self.conv.prepare(hidden)
        modified, _ = self.conv.prepare(changed)

        self.assertTrue(torch.equal(modified[:, :2], baseline[:, :2]))
        self.assertFalse(torch.equal(modified[:, 2], baseline[:, 2]))
        self.assertTrue(torch.equal(modified[:, 3:], baseline[:, 3:]))


class TestCandidateSelector(unittest.TestCase):
    def test_scores_match_public_inference_lattice_rows(self):
        selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=2)
        with torch.no_grad():
            selector.predecessor_codebook.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            )
            selector.successor_codebook.copy_(
                torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
            )
            selector.hidden_projection.weight.copy_(torch.eye(2))

        hidden = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
        logits = torch.tensor([[[0.1, 0.7, 0.2, 0.4], [0.6, 0.3, 0.9, 0.2]]])
        anchor_ids = torch.tensor([1])

        unary, public_candidate_ids = torch.topk(logits, 2, dim=-1, sorted=False)
        public_predecessor_ids = torch.cat(
            (
                anchor_ids[:, None, None].expand(-1, 1, 2),
                public_candidate_ids[:, :-1],
            ),
            dim=1,
        )
        public_lattice = unary[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            selector.predecessor_codebook[public_predecessor_ids] * hidden[:, :, None],
            selector.successor_codebook[public_candidate_ids],
        )

        previous_indices = torch.zeros(1, dtype=torch.long)
        selected_tokens = []
        realized_rows = []
        for position in range(hidden.shape[1]):
            row = public_lattice[:, position].gather(
                1,
                previous_indices[:, None, None].expand(-1, 1, selector.top_k),
            )[:, 0]
            previous_indices = row.argmax(dim=-1)
            selected_tokens.append(
                public_candidate_ids[:, position].gather(-1, previous_indices[:, None])[:, 0]
            )
            realized_rows.append(row)
        selected_tokens = torch.stack(selected_tokens, dim=1)
        predecessor_ids = torch.cat((anchor_ids[:, None], selected_tokens[:, :-1]), dim=1)

        scores, candidate_ids = selector.score_candidates(hidden, logits, predecessor_ids)

        self.assertTrue(torch.equal(candidate_ids, public_candidate_ids))
        self.assertTrue(torch.allclose(scores, torch.stack(realized_rows, dim=1), atol=1e-6))

    def test_training_candidates_replace_the_weakest_candidate_with_gold(self):
        selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=2)
        hidden = torch.ones(1, 1, 2)
        logits = torch.tensor([[[0.1, 0.9, 0.8, 0.2]]])

        _, candidate_ids = selector.score_candidates(
            hidden,
            logits,
            predecessor_ids=torch.tensor([[3]]),
            training_successor_ids=torch.tensor([[0]]),
        )

        self.assertEqual(set(candidate_ids[0, 0].tolist()), {0, 1})

        present_selector = CandidateSelector(hidden_size=2, vocab_size=4, rank=2, top_k=3)
        top_k = torch.topk(logits, 3, dim=-1, sorted=False)
        weakest_index = top_k.values.argmin(dim=-1).item()
        gold_index = 1 if weakest_index != 1 else 2
        gold_id = top_k.indices[..., gold_index]
        _, present_candidate_ids = present_selector.score_candidates(
            hidden,
            logits,
            predecessor_ids=torch.tensor([[3]]),
            training_successor_ids=gold_id,
        )
        self.assertEqual(
            set(present_candidate_ids[0, 0].tolist()),
            set(top_k.indices[0, 0].tolist()),
        )
        self.assertEqual(present_candidate_ids[0, 0].unique().numel(), 3)


class TestDFlash2Forward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = _make_model()

    def test_selector_loss_uses_teacher_forced_transitions_and_prefix_mask(self):
        config = _make_config(
            hidden_size=4,
            vocab_size=4,
            num_target_layers=1,
            selector_rank=1,
            selector_top_k=2,
        )
        model = DFlash2Model(
            DFlash2DraftModel(config),
            block_size=4,
            num_anchors=1,
            selector_loss_alpha=0.4,
        )
        selector = model.draft_model.candidate_selector
        with torch.no_grad():
            selector.hidden_projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            selector.predecessor_codebook.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
            selector.successor_codebook.copy_(torch.tensor([[0.5], [1.0], [1.5], [2.0]]))

        hidden = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 3.0, 2.0],
                    [0.1, 3.0, 0.0, 2.0],
                    [3.0, 2.0, 1.0, 0.0],
                ]
            ]
        )
        targets = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 0.5, 0.0]]])

        extra_numerator, components = model._extra_training_loss(hidden, logits, targets, weights)

        first_ce = torch.logsumexp(torch.tensor([4.5, 1.0]), dim=0) - 1.0
        second_ce = torch.logsumexp(torch.tensor([7.0, 6.0]), dim=0) - 6.0
        expected_numerator = first_ce + 0.5 * second_ce
        self.assertTrue(torch.allclose(extra_numerator, 0.4 * expected_numerator, atol=1e-6))
        component_numerator, component_denominator = components["selector_loss"]
        self.assertTrue(torch.allclose(component_numerator, expected_numerator, atol=1e-6))
        self.assertEqual(component_denominator.item(), 1.5)

        gap_weights = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
        gap_numerator, _ = model._extra_training_loss(hidden, logits, targets, gap_weights)
        self.assertTrue(torch.allclose(gap_numerator, 0.4 * first_ce, atol=1e-6))

    def test_gradients_reach_convolutions_and_selector(self):
        torch.manual_seed(13)
        result = self.model(**_batch(seed=2))
        self.assertEqual(len(result), 7)
        loss = result[0]
        loss_numerator, loss_denominator = result[-1]
        self.assertTrue(torch.allclose(loss, loss_numerator / loss_denominator.clamp(min=1e-6)))
        loss.backward()

        draft = self.model.draft_model

        def has_gradient(parameters):
            return any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in parameters
            )

        self.assertTrue(has_gradient(draft.layers[0].attention_conv.parameters()))
        self.assertTrue(has_gradient(draft.layers[0].mlp_conv.parameters()))
        self.assertGreater(draft.layers[0].attention_conv.base_kernel.grad[1].abs().sum(), 0)
        self.assertGreater(draft.layers[0].mlp_conv.base_kernel.grad[1].abs().sum(), 0)
        attention_projection_grad = draft.layers[0].attention_conv.kernel_projection.weight.grad
        mlp_projection_grad = draft.layers[0].mlp_conv.kernel_projection.weight.grad
        projection_half = attention_projection_grad.shape[0] // 2
        self.assertGreater(attention_projection_grad[projection_half:].abs().sum(), 0)
        self.assertGreater(mlp_projection_grad[projection_half:].abs().sum(), 0)
        self.assertTrue(has_gradient([draft.candidate_selector.predecessor_codebook]))
        self.assertTrue(has_gradient([draft.candidate_selector.successor_codebook]))
        self.assertTrue(has_gradient(draft.candidate_selector.hidden_projection.parameters()))
        self.assertTrue(has_gradient(draft.context_proj.parameters()))
        self.assertIsNone(draft.embed_tokens.weight.grad)

    def test_all_masked_batch_has_zero_losses(self):
        loss, _, _, _, _, components, loss_terms = self.model(**_batch(all_masked=True))

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(loss_terms[0].item(), 0.0)
        self.assertEqual(loss_terms[1].item(), 0.0)
        selector_numerator, selector_denominator = components["selector_loss"]
        self.assertEqual(selector_numerator.item(), 0.0)
        self.assertEqual(selector_denominator.item(), 0.0)

    def test_logit_scale_and_softcap_are_applied(self):
        self.model.draft_model.config.output_multiplier = 2.0
        self.model.draft_model.config.final_logit_softcapping = 0.5
        hidden = torch.tensor([[[1.0] * 16]])
        lm_head = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) / 100

        logits = self.model._compute_logits(hidden, lm_head)

        linear = F.linear(hidden, lm_head) * 2.0
        self.assertTrue(torch.allclose(logits, torch.tanh(linear / 0.5) * 0.5))

    def test_selector_loss_invalid_weights(self):
        _make_model(selector_loss_alpha=0.0)
        for alpha in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "must be non-negative"):
                    _make_model(selector_loss_alpha=alpha)

    def test_causal_sliding_mask_truth_table(self):
        mask = _create_dflash_mask_mod(
            anchor_positions=torch.tensor([[5]]),
            block_keep_mask=torch.tensor([[True]]),
            ctx_len=8,
            block_size=4,
            is_causal=True,
            sliding_window=3,
        )

        self.assertTrue(mask(0, 0, 0, 4))
        self.assertFalse(mask(0, 0, 2, 4))
        self.assertTrue(mask(0, 0, 2, 8))
        self.assertTrue(mask(0, 0, 2, 10))
        self.assertFalse(mask(0, 0, 2, 11))
