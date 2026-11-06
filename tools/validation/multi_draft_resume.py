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

"""Compare uninterrupted native training with a checkpoint-resumed native run.

No RNG override or reference weights are injected into the resumed training.
Coordinate comparisons run offline, after both independent trajectories finish.
"""

import argparse
import json
import math
from pathlib import Path


def tensor_difference(reference, actual, *, device="cpu"):
    import torch

    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("comparison tensor shape or dtype differs")
    total = [0.0, 0.0, 0.0, 0, 0]
    reference, actual = reference.reshape(-1), actual.reshape(-1)
    for offset in range(0, reference.numel(), 1024 * 1024):
        first = reference[offset : offset + 1024 * 1024].to(device=device, dtype=torch.float64)
        second = actual[offset : offset + 1024 * 1024].to(device=device, dtype=torch.float64)
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("comparison contains nonfinite coordinates")
        difference = second - first
        total[0] += float(first.square().sum())
        total[1] += float(difference.square().sum())
        total[2] = max(total[2], float(difference.abs().max()))
        total[3] += int((first != second).sum())
        total[4] += first.numel()
    return {
        "reference_l2": math.sqrt(total[0]),
        "difference_l2": math.sqrt(total[1]),
        "relative_l2": math.sqrt(total[1] / max(total[0], 1e-30)),
        "max_absolute": total[2],
        "different_coordinates": total[3],
        "coordinates": total[4],
    }


def compare_parameters(reference, actual, *, initial=False, device="cpu"):
    if reference.keys() != actual.keys():
        raise ValueError("comparison parameter names differ")
    totals = {}
    for name, fields in actual.items():
        expected_fields = set(reference[name]) - ({"gradient"} if initial else set())
        if set(fields) - ({"gradient"} if initial else set()) != expected_fields:
            raise ValueError(f"comparison state fields differ: {name}")
        for field, value in fields.items():
            if initial and field == "gradient":
                continue
            expected = reference[name][field]
            if value is None or expected is None:
                if value is not expected:
                    raise ValueError(f"optimizer state presence differs: {name}/{field}")
                continue
            row = tensor_difference(expected, value, device=device)
            total = totals.setdefault(
                field,
                {
                    "reference_squared": 0.0,
                    "difference_squared": 0.0,
                    "max_absolute": 0.0,
                    "different_coordinates": 0,
                    "coordinates": 0,
                    "worst_parameter": {"relative_l2": -1},
                },
            )
            total["reference_squared"] += row["reference_l2"] ** 2
            total["difference_squared"] += row["difference_l2"] ** 2
            total["max_absolute"] = max(total["max_absolute"], row["max_absolute"])
            total["different_coordinates"] += row["different_coordinates"]
            total["coordinates"] += row["coordinates"]
            if row["relative_l2"] > total["worst_parameter"]["relative_l2"]:
                total["worst_parameter"] = {"name": name, **row}
    for total in totals.values():
        total["reference_l2"] = math.sqrt(total.pop("reference_squared"))
        total["difference_l2"] = math.sqrt(total.pop("difference_squared"))
        total["relative_l2"] = total["difference_l2"] / max(total["reference_l2"], 1e-15)
    return totals


def compare_resume(
    reference_root,
    resumed_root,
    *,
    checkpoint_step,
    minimum_steps=16,
    device="cpu",
    initial_only=False,
):
    import torch

    reference_root, resumed_root = Path(reference_root), Path(resumed_root)
    source = reference_root / "replay-capture"
    resumed = resumed_root / "replay-capture"
    boundary = torch.load(source / f"state-{checkpoint_step:04d}.pt", weights_only=True)
    initial = torch.load(resumed / "initial-state.pt", weights_only=True)
    initial_diff = compare_parameters(
        boundary["parameters"], initial["parameters"], initial=True, device=device
    )
    if any(row["different_coordinates"] for row in initial_diff.values()):
        raise ValueError("checkpoint initial model/master/optimizer is not bitwise restored")
    if boundary["scheduler"] != initial["scheduler"]:
        raise ValueError("checkpoint scheduler was not exactly restored")
    initial_rng = {
        field: torch.equal(boundary[field], initial[field]) for field in ("torch_rng", "cuda_rng")
    }
    if not all(initial_rng.values()):
        raise ValueError("checkpoint initial RNG was not exactly restored")
    initial_result = {
        "initial_state_bitwise_restored": True,
        "initial_rng_equal": initial_rng,
        "initial_parameter_differences": initial_diff,
        "checkpoint_step": checkpoint_step,
        "diagnostic_only": True,
    }
    if initial_only:
        return initial_result
    del boundary, initial
    reference_rows = {
        row["step"]: row
        for row in map(json.loads, (source / "updates.jsonl").read_text().splitlines())
    }
    resumed_rows = list(map(json.loads, (resumed / "updates.jsonl").read_text().splitlines()))
    steps = [row["step"] for row in resumed_rows]
    if len(steps) < minimum_steps or steps != list(
        range(checkpoint_step + 1, checkpoint_step + 1 + len(steps))
    ):
        raise ValueError("require at least sixteen contiguous post-checkpoint optimizer steps")
    comparisons = []
    for row in resumed_rows:
        step = row["step"]
        expected = reference_rows[step]
        if not all(
            math.isfinite(item[key])
            for item in (row, expected)
            for key in ("loss", "gradient_norm", "learning_rate")
        ):
            raise ValueError("comparison contains nonfinite training metrics")
        if row["learning_rate"] != expected["learning_rate"]:
            raise ValueError("resumed learning rate differs from uninterrupted schedule")
        actual_inputs = sorted(resumed.glob(f"step-{step:04d}-micro-*.pt"))
        expected_inputs = sorted(source.glob(f"step-{step:04d}-micro-*.pt"))
        if not actual_inputs or [p.name for p in actual_inputs] != [
            p.name for p in expected_inputs
        ]:
            raise ValueError("resumed accumulation window differs")
        input_differences, rng_equal = {}, {"torch": True, "cuda": True}
        execution_ids_equal = True
        for first_path, second_path in zip(expected_inputs, actual_inputs, strict=True):
            first, second = [
                torch.load(path, weights_only=True) for path in (first_path, second_path)
            ]

            def logical(values):
                return [
                    {key: value for key, value in record.items() if key != "execution_id"}
                    for record in values
                ]

            if first.keys() != second.keys() or logical(first["_aspec_work"]) != logical(
                second["_aspec_work"]
            ):
                raise ValueError("resumed logical work differs from uninterrupted trajectory")
            execution_ids_equal &= first["_aspec_work"] == second["_aspec_work"]
            for field, value in second.items():
                if field == "_aspec_work":
                    continue
                if field.startswith("_audit"):
                    rng_equal["cuda" if "cuda" in field else "torch"] &= torch.equal(
                        first[field], value
                    )
                else:
                    diff = tensor_difference(first[field], value, device=device)
                    previous = input_differences.setdefault(
                        field, {"max_absolute": 0.0, "max_relative_l2": 0.0}
                    )
                    previous["max_absolute"] = max(previous["max_absolute"], diff["max_absolute"])
                    previous["max_relative_l2"] = max(
                        previous["max_relative_l2"], diff["relative_l2"]
                    )
        first, second = [
            torch.load(directory / f"state-{step:04d}.pt", weights_only=True)
            for directory in (source, resumed)
        ]
        states = compare_parameters(first["parameters"], second["parameters"], device=device)
        if first["scheduler"] != second["scheduler"] or states["step"]["different_coordinates"]:
            raise ValueError("resumed scheduler or Adam step diverged")
        loss_abs = abs(row["loss"] - expected["loss"])
        norm_abs = abs(row["gradient_norm"] - expected["gradient_norm"])
        comparisons.append(
            {
                "step": step,
                "input_differences": input_differences,
                "execution_ids_equal": execution_ids_equal,
                "batch_rng_equal": rng_equal,
                "states": states,
                "loss_absolute_diff": loss_abs,
                "loss_relative_diff": loss_abs / max(abs(expected["loss"]), 1e-12),
                "gradient_norm_absolute_diff": norm_abs,
                "gradient_norm_relative_diff": norm_abs
                / max(abs(expected["gradient_norm"]), 1e-12),
            }
        )
    return {
        "diagnostic_only": True,
        "checkpoint_step": checkpoint_step,
        "steps": len(steps),
        "initial_state_bitwise_restored": True,
        "initial_rng_equal": initial_rng,
        "rng_overridden_during_training": False,
        "semantics_precision_lr_changed": False,
        "comparisons": comparisons,
    }


def main():
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    result = compare_resume(
        args.reference,
        args.resumed,
        checkpoint_step=args.checkpoint_step,
        device=args.device,
        initial_only=args.initial_only,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if args.initial_only:
        print(
            json.dumps(
                {
                    "initial_state_bitwise_restored": True,
                    "initial_rng_equal": result["initial_rng_equal"],
                }
            )
        )
        return
    print(
        json.dumps(
            {
                "steps": result["steps"],
                "initial_state_bitwise_restored": True,
                "max_loss_relative_diff": max(
                    row["loss_relative_diff"] for row in result["comparisons"]
                ),
                "batch_rng_equal": all(
                    all(row["batch_rng_equal"].values()) for row in result["comparisons"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
