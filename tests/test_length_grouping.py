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
# tests/test_length_grouping.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for length grouping."""

from dataclasses import dataclass
from queue import Queue
from unittest.mock import patch

import pytest
import torch

from aspec.controller.training_controller import (
    AsyncTrainingController,
    dataset_fingerprint,
)
from aspec.data.utils import length_grouped_order, sample_length_hint
from aspec.work_lifecycle import WORK_ID_KEY, build_work_identity


def samples(*lengths):
    return [{"data_id": f"s{i}", "seq_len": n} for i, n in enumerate(lengths)]


def lengths_of(entries):
    return [sample_length_hint(e) for e in entries]


@pytest.mark.parametrize(
    "sample, expected",
    [
        ({"seq_len": 512}, 512),  # offline replay row
        ({"input_ids": torch.zeros(128), "formatted_prompt": "ignored"}, 128),  # tokenized
        ({"formatted_prompt": "hello"}, 5),  # defer_tokenization: chars stand in
    ],
)
def test_length_hint_reads_whichever_field_is_present(sample, expected):
    assert sample_length_hint(sample) == expected


def test_sorts_longest_first_within_each_chunk():
    # Chunks are positional, so the 100 in the second chunk stays there.
    assert lengths_of(length_grouped_order(samples(1, 2, 100, 3), 2)) == [2, 1, 100, 3]


def test_trailing_partial_chunk_is_sorted_too():
    assert lengths_of(length_grouped_order(samples(1, 2, 3, 4, 9, 5), 4)) == [4, 3, 2, 1, 9, 5]


def test_small_dataset_becomes_a_full_sort():
    assert lengths_of(length_grouped_order(samples(1, 5, 3), 1024)) == [5, 3, 1]


@pytest.mark.parametrize("group_size", [0, 1])
def test_group_size_of_one_or_less_disables_grouping(group_size):
    entries = samples(1, 5, 3)
    assert length_grouped_order(entries, group_size) == entries


def test_leaves_the_input_list_untouched():
    entries = samples(1, 5, 3)
    length_grouped_order(entries, 3)
    assert lengths_of(entries) == [1, 5, 3]


def test_groups_similar_lengths_into_neighbouring_dispatches():
    # The property that matters: a dispatch-sized window should span a narrow
    # range of lengths after grouping.
    entries = samples(*((i * 37) % 1000 for i in range(600)))

    def mean_spread(ordered, dispatch=4):
        windows = [lengths_of(ordered[i : i + dispatch]) for i in range(0, 600, dispatch)]
        return sum(max(w) - min(w) for w in windows) / len(windows)

    assert mean_spread(length_grouped_order(entries, 128)) < mean_spread(entries) / 5


# --- controller wiring ------------------------------------------------------


@dataclass
class MockArgs:
    per_dp_rank_batch_size: int = 1
    max_sample_pool_size: int = 0
    seed: int = 0
    shuffle_dataset: bool = True
    length_group_size: int = 8


def controller(dataset, **overrides):
    controller_class = AsyncTrainingController.__ray_metadata__.modified_class
    with patch("aspec.controller.training_controller.Queue", Queue):
        actor = controller_class(MockArgs(**overrides), dp_size=2)
    actor._stored_dataset = dataset
    return actor


SHUFFLE_ME = samples(*((i * 13) % 100 for i in range(32)))


def test_grouping_runs_after_the_shuffle():
    data = controller(SHUFFLE_ME)._prepare_dataset()

    assert len(data) == 32
    for start in range(0, 32, 8):
        chunk = lengths_of(data[start : start + 8])
        assert chunk == sorted(chunk, reverse=True)


def test_disabled_grouping_leaves_the_shuffled_order():
    grouped = controller(SHUFFLE_ME)._prepare_dataset()
    plain = controller(SHUFFLE_ME, length_group_size=0)._prepare_dataset()
    assert lengths_of(grouped) != lengths_of(plain)


def test_same_epoch_rebuilds_the_same_order():
    # Mid-epoch resume slices this order, so it has to be reproducible.
    first = controller(SHUFFLE_ME)._prepare_dataset()
    second = controller(SHUFFLE_ME)._prepare_dataset()
    assert [e["data_id"] for e in first] == [e["data_id"] for e in second]


def test_skip_slices_the_grouped_order():
    actor = controller(SHUFFLE_ME)
    assert actor._prepare_dataset(skip=10) == actor._prepare_dataset()[10:]


def test_exact_resume_filters_committed_work_identity_not_a_prefix():
    actor = controller(SHUFFLE_ME)
    committed_id = "s7"
    committed = {
        build_work_identity(split="train", epoch=0, source_id=committed_id)[WORK_ID_KEY]
    }

    remaining = actor._prepare_dataset(committed_work_ids=committed)

    assert committed_id not in {sample["data_id"] for sample in remaining}
    assert len(remaining) == len(SHUFFLE_ME) - 1


def test_dataset_fingerprint_covers_row_content_and_identity():
    first = [
        {
            "data_id": "s0",
            "input_ids": torch.tensor([1, 2, 3]),
            "packed_loss_mask": torch.tensor([7], dtype=torch.uint8),
        }
    ]
    second = [
        {
            "data_id": "s0",
            "input_ids": torch.tensor([1, 2, 4]),
            "packed_loss_mask": torch.tensor([7], dtype=torch.uint8),
        }
    ]

    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_dataset_fingerprint_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate dataset data_id"):
        dataset_fingerprint([{"data_id": "same"}, {"data_id": "same"}])


def test_each_epoch_groups_different_samples_together():
    actor = controller(SHUFFLE_ME)
    epoch0 = actor._prepare_dataset()
    actor._dataset_epoch = 1
    epoch1 = actor._prepare_dataset()

    assert {e["data_id"] for e in epoch0[:8]} != {e["data_id"] for e in epoch1[:8]}


def test_profile_source_restart_drops_partial_epoch_and_replays_full_dataset():
    actor = controller(SHUFFLE_ME)
    assert actor.submit_training_dataset() == len(SHUFFLE_ME)
    profiled = actor.get_prompts(5)

    result = actor.restart_training_source_after_profile()
    replayed = actor.get_prompts(len(SHUFFLE_ME) + 1)

    assert result == {
        "state": "reloaded",
        "discarded_prompts": len(SHUFFLE_ME) - len(profiled),
        "reloaded_samples": len(SHUFFLE_ME),
        "dataset_epoch": 1,
    }
    assert len(replayed) == len(SHUFFLE_ME)
    assert {
        entry.metadata[WORK_ID_KEY] for entry in profiled
    }.isdisjoint({entry.metadata[WORK_ID_KEY] for entry in replayed})
