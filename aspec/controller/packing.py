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

"""Deterministic, work-aware row packing for the live training path.

Adapted from AngelSpec's fixed-row packing helpers.  Rows are seeded with the
oldest work before later candidates are reordered, so optimization cannot
starve an awkward old sample.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackingResult:
    rank_rows: list[list[int]]
    selected_indices: tuple[int, ...]
    packed_tokens: int
    capacity_tokens: int
    useful_anchor_starts: int

    @property
    def fill_ratio(self) -> float:
        return self.packed_tokens / self.capacity_tokens if self.capacity_tokens else 0.0


def count_eligible_anchor_starts(
    packed_loss_mask: str | list[int], sample_length: int, block_size: int
) -> int:
    """Count supervised anchor starts whose complete block remains in a document."""
    if sample_length <= 0 or block_size <= 0:
        return 0
    if isinstance(packed_loss_mask, str):
        try:
            runs = [int(value) for value in packed_loss_mask.split(",") if value]
        except ValueError as exc:
            raise ValueError("packed_loss_mask contains a non-integer run length") from exc
    else:
        runs = [int(value) for value in packed_loss_mask]
    if any(length < 0 for length in runs):
        raise ValueError("packed_loss_mask run lengths must be non-negative")

    eligible_prefix = max(sample_length - block_size + 1, 0)
    cursor = 0
    count = 0
    for index, run_length in enumerate(runs):
        run_end = cursor + run_length
        if index % 2 == 1:
            count += max(0, min(run_end, eligible_prefix) - min(cursor, eligible_prefix))
        cursor = run_end
    return count


def count_eagle_supervised_positions(packed_loss_mask, sample_length, ttt_length):
    """Count real next-token supervision over the recurrent TTT depths."""
    runs = ([int(v) for v in packed_loss_mask.split(",") if v]
            if isinstance(packed_loss_mask, str) else list(packed_loss_mask))
    if any(v < 0 for v in runs):
        raise ValueError("packed_loss_mask run lengths must be non-negative")
    cursor = total = 0
    for index, length in enumerate(runs):
        end = cursor + length
        if index % 2:
            total += sum(max(0, min(end, sample_length - 1) - max(cursor, depth))
                         for depth in range(ttt_length))
        cursor = end
    return total


def pack_work_aware_rows(
    lengths: list[int],
    useful_work: list[int],
    *,
    dp_size: int,
    max_seq_length: int,
    max_selected: int | None = None,
) -> PackingResult | None:
    """Build one row per DP rank with oldest-work progress and best-fit reuse.

    Every row is seeded from the oldest ``dp_size`` samples. Remaining samples
    are considered by useful-anchor density, then placed in the tightest row
    they fit. The result therefore optimizes learner work without making token
    fill the sole objective.
    """
    if len(lengths) != len(useful_work):
        raise ValueError("lengths and useful_work must have the same size")
    if dp_size <= 0 or max_seq_length <= 0:
        raise ValueError("dp_size and max_seq_length must be positive")
    if max_selected is not None and max_selected < dp_size:
        raise ValueError("max_selected must be at least dp_size")
    invalid = [
        (index, length)
        for index, length in enumerate(lengths)
        if length <= 0 or length > max_seq_length
    ]
    if invalid:
        preview = ", ".join(f"{index}:{length}" for index, length in invalid[:5])
        raise ValueError(
            "Packing received work that can never fit a row: "
            f"{preview}; max_seq_length={max_seq_length}"
        )
    if len(lengths) < dp_size:
        return None

    rows = [[index] for index in range(dp_size)]
    row_tokens = lengths[:dp_size].copy()
    selected = set(range(dp_size))
    candidates = sorted(
        range(dp_size, len(lengths)),
        key=lambda index: (
            -(useful_work[index] / lengths[index]),
            -useful_work[index],
            -lengths[index],
            index,
        ),
    )
    for index in candidates:
        if max_selected is not None and len(selected) >= max_selected:
            break
        fitting = [
            row_index
            for row_index, used in enumerate(row_tokens)
            if used + lengths[index] <= max_seq_length
        ]
        if not fitting:
            continue
        row_index = min(
            fitting,
            key=lambda candidate: (
                max_seq_length - row_tokens[candidate] - lengths[index],
                row_tokens[candidate],
                candidate,
            ),
        )
        rows[row_index].append(index)
        row_tokens[row_index] += lengths[index]
        selected.add(index)

    # Longest rows go to the currently lightest rank. There is one row/rank in
    # this API, so this mainly makes the assignment deterministic across runs.
    order = sorted(range(dp_size), key=lambda index: (-row_tokens[index], index))
    rank_rows = [rows[index] for index in order]
    selected_indices = tuple(sorted(selected))
    return PackingResult(
        rank_rows=rank_rows,
        selected_indices=selected_indices,
        packed_tokens=sum(lengths[index] for index in selected_indices),
        capacity_tokens=dp_size * max_seq_length,
        useful_anchor_starts=sum(useful_work[index] for index in selected_indices),
    )
