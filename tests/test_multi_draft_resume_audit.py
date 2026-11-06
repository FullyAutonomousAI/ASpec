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

import json

import pytest
import torch

from tools.validation.multi_draft_resume import compare_resume, tensor_difference


def _captured(tmp_path):
    first, second = tmp_path / "reference", tmp_path / "resumed"
    for root in (first, second):
        (root / "replay-capture").mkdir(parents=True)

    def state(step):
        return {
            "parameters": {
                "weight": {
                    "model": torch.tensor([1.0, 2.0]),
                    "master": torch.tensor([1.0, 2.0]),
                    "exp_avg": torch.tensor([0.1, 0.2]),
                    "exp_avg_sq": torch.tensor([0.01, 0.04]),
                    "step": torch.tensor(float(step)),
                    "gradient": torch.tensor([0.3, 0.4]),
                }
            },
            "scheduler": {"last_epoch": step},
            "torch_rng": torch.tensor([1, 2], dtype=torch.uint8),
            "cuda_rng": torch.tensor([3, 4], dtype=torch.uint8),
        }

    initial = state(4)
    torch.save(initial, first / "replay-capture/state-0004.pt")
    initial["parameters"]["weight"].pop("gradient")
    torch.save(initial, second / "replay-capture/initial-state.pt")
    rows = [
        {"step": step, "loss": 2.0, "gradient_norm": 0.5, "learning_rate": 0.001}
        for step in range(5, 21)
    ]
    for root in (first, second):
        capture = root / "replay-capture"
        (capture / "updates.jsonl").write_text("\n".join(map(json.dumps, rows)))
        for row in rows:
            step = row["step"]
            torch.save(state(step), capture / f"state-{step:04d}.pt")
            torch.save(
                {
                    "input_ids": torch.tensor([step]),
                    "hidden_states": torch.tensor([1.0]),
                    "_aspec_work": [{"work_id": str(step), "execution_id": f"{step}:attempt:0"}],
                    "_audit_rng": initial["torch_rng"],
                    "_audit_cuda_rng": initial["cuda_rng"],
                },
                capture / f"step-{step:04d}-micro-00.pt",
            )
    return first, second


def test_native_resume_audit_checks_sixteen_full_coordinate_updates(tmp_path):
    reference, resumed = _captured(tmp_path)
    result = compare_resume(reference, resumed, checkpoint_step=4)
    assert result["steps"] == 16
    assert result["initial_state_bitwise_restored"]
    assert not result["rng_overridden_during_training"]
    assert all(
        row["states"]["gradient"]["different_coordinates"] == 0 for row in result["comparisons"]
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("initial", "bitwise restored"),
        ("initial_rng", "initial RNG"),
        ("work", "logical work"),
        ("lr", "learning rate"),
        ("adam_step", "Adam step"),
        ("missing_step", "contiguous"),
    ],
)
def test_native_resume_audit_rejects_semantic_or_restoration_mismatch(tmp_path, mutation, match):
    reference, resumed = _captured(tmp_path)
    capture = resumed / "replay-capture"
    if mutation in {"initial", "initial_rng"}:
        path = capture / "initial-state.pt"
        state = torch.load(path, weights_only=True)
        if mutation == "initial":
            state["parameters"]["weight"]["master"][0] += 1
        else:
            state["cuda_rng"][0] += 1
        torch.save(state, path)
    elif mutation == "work":
        path = capture / "step-0005-micro-00.pt"
        batch = torch.load(path, weights_only=True)
        batch["_aspec_work"][0]["work_id"] = "foreign"
        torch.save(batch, path)
    elif mutation == "adam_step":
        path = capture / "state-0005.pt"
        state = torch.load(path, weights_only=True)
        state["parameters"]["weight"]["step"] += 1
        torch.save(state, path)
    else:
        path = capture / "updates.jsonl"
        rows = list(map(json.loads, path.read_text().splitlines()))
        if mutation == "lr":
            rows[0]["learning_rate"] *= 2
        else:
            rows.pop(0)
        path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match=match):
        compare_resume(reference, resumed, checkpoint_step=4)


def test_coordinate_diff_is_not_a_gradient_norm_proxy():
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    assert torch.linalg.vector_norm(first) == torch.linalg.vector_norm(second)
    result = tensor_difference(first, second)
    assert result["relative_l2"] == pytest.approx(2**0.5)
    assert result["max_absolute"] == 1
    assert result["different_coordinates"] == 2
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_difference(first, torch.tensor([float("nan"), 0.0]))


def test_resume_audit_preserves_semantic_work_without_pinning_attempt_identity(tmp_path):
    reference, resumed = _captured(tmp_path)
    path = resumed / "replay-capture/step-0005-micro-00.pt"
    batch = torch.load(path, weights_only=True)
    batch["_aspec_work"][0]["execution_id"] = "5:attempt:1"
    torch.save(batch, path)
    result = compare_resume(reference, resumed, checkpoint_step=4)
    assert not result["comparisons"][0]["execution_ids_equal"]


def test_initial_only_audit_does_not_claim_a_completed_resume_trajectory(tmp_path):
    reference, resumed = _captured(tmp_path)
    result = compare_resume(reference, resumed, checkpoint_step=4, initial_only=True)
    assert result["initial_state_bitwise_restored"]
    assert "comparisons" not in result
