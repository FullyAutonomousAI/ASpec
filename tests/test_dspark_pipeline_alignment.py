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

"""Cheap regressions for the full-chain replay protocol; no GPU or model assets."""

import json
from argparse import Namespace

import pytest
import torch

from tools.validation.dspark_pipeline_alignment import (
    compare,
    gradient_sample_indices,
    replay_anchors,
)


@pytest.mark.parametrize("numel", [1, 7, 622329856, 1126422017])
def test_gradient_witness_indices_do_not_round_outside_large_parameters(numel):
    indices = gradient_sample_indices(numel, "cpu")
    assert indices.dtype == torch.long
    assert indices[0] == 0
    assert indices[-1] == numel - 1
    assert len(indices.unique()) == min(512, numel)


def test_replay_ignores_row_padding_and_global_rng():
    docs = torch.tensor([[0] * 17 + [1] * 23])
    mask = torch.ones_like(docs, dtype=torch.float32)
    mask[:, [0, 17, 24]] = 0
    a, keep = replay_anchors(mask, docs, 12, 7, 731)
    torch.manual_seed(998)
    padded = replay_anchors(
        torch.nn.functional.pad(mask, (0, 24)),
        torch.nn.functional.pad(docs, (0, 24), value=-1),
        12,
        7,
        731,
    )
    assert torch.equal(a, padded[0])
    assert torch.equal(keep, padded[1])
    for position in a[keep].tolist():
        assert docs[0, position] == docs[0, position + 7]
        assert mask[0, position] == mask[0, position + 1] == 1


def test_empty_replay_keeps_inactive_slots_zero():
    anchors, keep = replay_anchors(torch.zeros(1, 3), torch.zeros(1, 3), 8, 7, 1)
    assert not keep.any()
    assert not anchors.any()


def test_replay_changes_with_seed_without_changing_validity():
    docs = torch.zeros(1, 100, dtype=torch.long)
    mask = torch.ones(1, 100)
    a = replay_anchors(mask, docs, 16, 7, 1)
    b = replay_anchors(mask, docs, 16, 7, 2)
    assert not torch.equal(a[0], b[0])
    assert a[1].all() and b[1].all()


def _write_fixture(root):
    for role in ("angel", "fast-cp1", "fast-cp2", "fast-cp4"):
        path = root / role
        path.mkdir()
        steps = [
            {
                "step": step,
                "loss": 2.0,
                "ce": 5.0,
                "l1": 1.0,
                "confidence": 0.6,
                "denominator": 10.0,
                "gradient_norm": 1.0,
            }
            for step in range(1, 17)
        ]
        traces = [
            {
                "step": step,
                "microbatch": micro,
                "document_lengths": [20, 30],
                "input_sha256": "ids",
                "mask_sha256": "mask",
                "anchors_sha256": "a",
                "keep_sha256": "keep",
                "feature_sha256": "features",
                "active_anchors": 4,
                "input_tokens": 50,
            }
            for step in range(1, 17)
            for micro in range(2)
        ]
        for name, records in (("loss-steps", steps), ("microbatches", traces)):
            (path / (name + ".jsonl")).write_text("\n".join(map(json.dumps, records)) + "\n")


def test_compare_requires_every_step_and_equal_features(tmp_path):
    _write_fixture(tmp_path)
    args = Namespace(run_dir=str(tmp_path), steps=16)
    compare(args)
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["fast-cp4"]["max_relative_diff"] == 0
    path = tmp_path / "fast-cp2" / "microbatches.jsonl"
    path.write_text(
        path.read_text().replace('"feature_sha256": "features"', '"feature_sha256": "different"', 1)
    )
    with pytest.raises(ValueError, match="input/replay mismatch"):
        compare(args)


def _write_trajectory(root):
    for role in ("angel", "fast-cp1", "fast-cp2", "fast-cp4"):
        records = [
            {
                "step": step,
                "state_sha256": f"state-{step}",
                "mode": "capture" if role == "angel" else "replay",
            }
            for step in range(1, 17)
        ]
        (root / role / "weight-trajectory.jsonl").write_text(
            "\n".join(map(json.dumps, records)) + "\n"
        )


def test_common_weight_comparison_requires_the_actual_same_states(tmp_path):
    _write_fixture(tmp_path)
    _write_trajectory(tmp_path)
    args = Namespace(run_dir=str(tmp_path), steps=16)
    compare(args)
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["protocol"] == "common-weight-replay"
    path = tmp_path / "fast-cp4" / "weight-trajectory.jsonl"
    path.write_text(path.read_text().replace('"state-16"', '"wrong-state"', 1))
    with pytest.raises(ValueError, match="common-weight trajectory mismatch"):
        compare(args)


@pytest.mark.parametrize("failure", ["missing", "incomplete", "wrong-mode"])
def test_common_weight_comparison_rejects_partial_or_mixed_protocols(tmp_path, failure):
    _write_fixture(tmp_path)
    _write_trajectory(tmp_path)
    path = tmp_path / "fast-cp2" / "weight-trajectory.jsonl"
    if failure == "missing":
        path.unlink()
    elif failure == "incomplete":
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    else:
        path.write_text(path.read_text().replace('"replay"', '"capture"'))
    with pytest.raises(ValueError, match="common-weight trajectory"):
        compare(Namespace(run_dir=str(tmp_path), steps=16))


def test_comparison_rejects_nonfinite_observations(tmp_path):
    _write_fixture(tmp_path)
    path = tmp_path / "fast-cp1" / "loss-steps.jsonl"
    path.write_text(path.read_text().replace('"loss": 2.0', '"loss": NaN', 1))
    with pytest.raises(ValueError, match="non-finite"):
        compare(Namespace(run_dir=str(tmp_path), steps=16))
