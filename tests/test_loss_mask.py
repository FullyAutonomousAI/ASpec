# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_loss_mask.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for compute_assistant_loss_mask: correctness across token sequence shapes and edge cases."""

import pytest
import torch

from aspec.models.ops.loss_mask import compute_assistant_loss_mask


def _reference_loss_mask(input_ids, assistant_header_ids, end_token_ids):
    """Reference implementation using Python loops for correctness checking."""
    header_len = len(assistant_header_ids)
    end_len = len(end_token_ids)

    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    loss_mask = torch.zeros(len(ids), dtype=torch.long)

    i = 0
    while i < len(ids) - header_len + 1:
        if ids[i : i + header_len] == assistant_header_ids:
            j = i + header_len
            while j <= len(ids) - end_len:
                if ids[j : j + end_len] == end_token_ids:
                    break
                loss_mask[j] = 1
                j += 1
            else:
                for k in range(j, len(ids)):
                    loss_mask[k] = 1
            i = j + end_len
        else:
            i += 1

    return loss_mask


def _check(input_ids, header_ids, end_ids, expected):
    ids = torch.tensor(input_ids, dtype=torch.long)
    expected_t = torch.tensor(expected, dtype=torch.long)

    ref = _reference_loss_mask(ids, header_ids, end_ids)
    result = compute_assistant_loss_mask(ids, header_ids, end_ids)

    assert torch.equal(ref, expected_t), f"reference mismatch: {ref.tolist()} != {expected}"
    assert torch.equal(result, expected_t), f"numba mismatch: {result.tolist()} != {expected}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def _check_gpu(input_ids, header_ids, end_ids, expected):
    ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
    expected_t = torch.tensor(expected, dtype=torch.long)
    result = compute_assistant_loss_mask(ids, header_ids, end_ids)
    assert torch.equal(result.cpu(), expected_t), (
        f"GPU mismatch: {result.cpu().tolist()} != {expected}"
    )


# ── Multi-token header, multi-token end ──────────────────────────────


class TestMultiTokenHeaderMultiTokenEnd:
    H = [10, 20]
    E = [30, 40]

    def test_single_turn(self):
        _check([10, 20, 1, 2, 3, 30, 40], self.H, self.E, [0, 0, 1, 1, 1, 0, 0])

    def test_two_turns(self):
        _check(
            [10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40],
            self.H,
            self.E,
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        )

    def test_no_end_token(self):
        _check([10, 20, 1, 2, 3], self.H, self.E, [0, 0, 1, 1, 1])

    def test_no_header(self):
        _check([5, 6, 7, 8], self.H, self.E, [0, 0, 0, 0])

    def test_empty_input(self):
        _check([], self.H, self.E, [])

    def test_header_immediately_followed_by_end(self):
        _check([10, 20, 30, 40], self.H, self.E, [0, 0, 0, 0])

    def test_second_turn_no_end(self):
        _check([10, 20, 1, 30, 40, 10, 20, 2, 3], self.H, self.E, [0, 0, 1, 0, 0, 0, 0, 1, 1])


# ── Single-token header, single-token end ────────────────────────────


class TestSingleTokenHeaderSingleTokenEnd:
    H = [10]
    E = [30]

    def test_single_turn(self):
        _check([10, 1, 2, 3, 30], self.H, self.E, [0, 1, 1, 1, 0])

    def test_two_turns(self):
        _check([10, 1, 30, 10, 2, 30], self.H, self.E, [0, 1, 0, 0, 1, 0])

    def test_no_end_token(self):
        _check([10, 1, 2, 3], self.H, self.E, [0, 1, 1, 1])

    def test_header_immediately_followed_by_end(self):
        _check([10, 30], self.H, self.E, [0, 0])


# ── Mixed: multi-token header + single-token end ────────────────────


class TestMultiTokenHeaderSingleTokenEnd:
    def test_two_turns(self):
        _check([10, 20, 1, 2, 30, 10, 20, 3, 30], [10, 20], [30], [0, 0, 1, 1, 0, 0, 0, 1, 0])


# ── Mixed: single-token header + multi-token end ────────────────────


class TestSingleTokenHeaderMultiTokenEnd:
    def test_two_turns(self):
        _check([10, 1, 2, 30, 40, 10, 3, 30, 40], [10], [30, 40], [0, 1, 1, 0, 0, 0, 1, 0, 0])


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    H = [10, 20]
    E = [30, 40]

    def test_input_is_exactly_header(self):
        _check([10, 20], self.H, self.E, [0, 0])

    def test_input_shorter_than_header(self):
        _check([10], self.H, [30], [0])

    def test_end_tokens_before_any_header(self):
        _check([30, 40, 30, 40, 10, 20, 1, 30, 40], self.H, self.E, [0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_extra_end_tokens_between_turns(self):
        _check(
            [10, 20, 1, 30, 40, 30, 40, 10, 20, 2, 30, 40],
            self.H,
            self.E,
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        )

    def test_long_assistant_content(self):
        _check(
            [10, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 40],
            self.H,
            self.E,
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        )

    def test_three_turns_all_empty_content(self):
        _check(
            [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40],
            self.H,
            self.E,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_mixed_content_empty_content(self):
        _check(
            [10, 20, 1, 30, 40, 10, 20, 30, 40, 10, 20, 2, 3, 30, 40],
            self.H,
            self.E,
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        )


# ── Long context (200k tokens) correctness ───────────────────────────


class TestLongContext:
    H = [128006, 77091, 128007, 271]
    E = [128009]

    @staticmethod
    def _make_input_ids(seq_len, num_turns, seed=42):
        rng = torch.Generator().manual_seed(seed)
        header = TestLongContext.H
        end = TestLongContext.E
        ids: list[int] = []
        content_len = max(1, (seq_len - num_turns * (len(header) + len(end))) // num_turns)
        for _ in range(num_turns):
            ids.extend(header)
            ids.extend(torch.randint(1000, 30000, (content_len,), generator=rng).tolist())
            ids.extend(end)
        if len(ids) < seq_len:
            ids.extend(torch.randint(1000, 30000, (seq_len - len(ids),), generator=rng).tolist())
        return torch.tensor(ids[:seq_len], dtype=torch.long)

    def test_200k_multiturn(self):
        ids = self._make_input_ids(200_000, num_turns=10)
        ref = _reference_loss_mask(ids, self.H, self.E)
        result = compute_assistant_loss_mask(ids, self.H, self.E)
        assert torch.equal(result, ref)

    def test_200k_no_header(self):
        rng = torch.Generator().manual_seed(99)
        ids = torch.randint(1000, 30000, (200_000,), generator=rng)
        result = compute_assistant_loss_mask(ids, self.H, self.E)
        assert result.sum() == 0

    def test_200k_trailing_open_turn(self):
        ids = self._make_input_ids(200_000, num_turns=5)
        ids_list = ids.tolist()
        ids_list.extend(self.H)
        ids_list.extend(list(range(1000, 1100)))
        ids_open = torch.tensor(ids_list, dtype=torch.long)
        ref = _reference_loss_mask(ids_open, self.H, self.E)
        result = compute_assistant_loss_mask(ids_open, self.H, self.E)
        assert torch.equal(result, ref)
        assert result[-100:].sum() == 100


# ── last_turn_only ───────────────────────────────────────────────────


class TestLastTurnOnly:
    H = [10, 20]
    E = [30, 40]

    def test_single_turn_unchanged(self):
        ids = torch.tensor([10, 20, 1, 2, 3, 30, 40], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == [0, 0, 1, 1, 1, 0, 0]

    def test_two_turns_keeps_last(self):
        ids = torch.tensor([10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_three_turns_keeps_last(self):
        ids = torch.tensor(
            [10, 20, 1, 30, 40, 10, 20, 2, 30, 40, 10, 20, 3, 4, 5, 30, 40], dtype=torch.long
        )
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

    def test_truncated_last_turn(self):
        """Sequence cut mid-response — last turn has no end token."""
        ids = torch.tensor([10, 20, 1, 30, 40, 10, 20, 2, 3], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]

    def test_no_assistant_turns(self):
        ids = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == [0, 0, 0, 0]

    def test_empty_input(self):
        ids = torch.tensor([], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=True)
        assert result.tolist() == []

    def test_false_flag_returns_all_turns(self):
        ids = torch.tensor([10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40], dtype=torch.long)
        result = compute_assistant_loss_mask(ids, self.H, self.E, last_turn_only=False)
        assert result.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize("tail", [[10, 20], [10, 20, 30, 40]])
def test_empty_final_assistant_turn_does_not_train_on_an_earlier_turn(tail):
    ids = torch.tensor([10, 20, 1, 2, 30, 40, *tail], dtype=torch.long)
    result = compute_assistant_loss_mask(ids, [10, 20], [30, 40], last_turn_only=True)
    assert not result.any()


def test_final_assistant_turn_with_only_skipped_header_token_has_no_loss():
    ids = torch.tensor([10, 20, 99, 1, 2, 30, 40, 10, 20, 99], dtype=torch.long)
    result = compute_assistant_loss_mask(
        ids, [10, 20], [30, 40], last_turn_only=True, skip_after_header=1
    )
    assert not result.any()
