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

"""Summarize one captured/replayed shared-source acceptance run."""

import argparse
import json
from pathlib import Path


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--replay-a", required=True)
    parser.add_argument("--replay-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.session)
    evidence = json.loads((root / "evidence.json").read_text())
    summaries = {
        name: json.loads((Path(path) / "summary.json").read_text())
        for name, path in (("draft_a", args.replay_a), ("draft_b", args.replay_b))
    }
    shared_inputs, shared_coordinates = 0, 0
    for path in sorted((root / "draft_a" / "replay-capture").glob("step-*-micro-*.pt")):
        first = torch.load(path, weights_only=True)
        second = torch.load(root / "draft_b" / "replay-capture" / path.name, weights_only=True)
        for key, tensor in first.items():
            if key.startswith("_audit"):
                continue
            if isinstance(tensor, torch.Tensor):
                if not torch.equal(tensor, second[key]):
                    raise RuntimeError(f"shared observation differs: {path.name}/{key}")
                shared_coordinates += tensor.numel()
            elif tensor != second[key]:
                raise RuntimeError(f"shared logical work differs: {path.name}/{key}")
        shared_inputs += 1
    result = {
        "diagnostic_only": True,
        "shared_feature_windows_bitwise_equal": shared_inputs,
        "shared_feature_coordinates_checked": shared_coordinates,
        "shared_source": evidence["shared_source"],
        "experiments": {},
    }
    for name, summary in summaries.items():
        rows = summary["comparisons"]
        exact = summary["all_states_equal"]
        result["experiments"][name] = {
            "steps": summary["steps"],
            "whole_state_bitwise_equal": exact,
            "loss_metric": "pooled_native_dspark_combined_objective",
            "maximum_loss_absolute_diff": max(row["loss_absolute_diff"] for row in rows),
            "maximum_loss_relative_diff": max(row["loss_relative_diff"] for row in rows),
            "maximum_gradient_norm_absolute_diff": max(
                row["gradient_norm_absolute_diff"] for row in rows
            ),
            "maximum_gradient_norm_relative_diff": max(
                row["gradient_norm_relative_diff"] for row in rows
            ),
            "gradient_model_master_moments_absolute_diff": 0 if exact else None,
            "gradient_model_master_moments_relative_diff": 0 if exact else None,
            "final_state_full_coordinate_differences": summary.get(
                "final_state_full_coordinate_differences"
            ),
        }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
