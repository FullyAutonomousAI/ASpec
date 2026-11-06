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

"""Inspect actual DSpark checkpoint weights/masters and native Adam update counts."""

import argparse
import json
from pathlib import Path

from tools.validation.dspark_nonpacking_alignment import vector_comparison


def read_group(directory, section, field, prefix):
    import torch
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(str(directory)).read_metadata()
    tensors = {
        key[len(prefix) :]: torch.empty(value.size, dtype=value.properties.dtype)
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith(prefix)
    }
    if not tensors:
        raise ValueError(f"No {prefix} tensors in {directory}")
    payload = {section: {field: tensors}}
    dcp.load(payload, checkpoint_id=str(directory))
    return payload[section][field]


def adam_counts(directory):
    import torch
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(str(directory)).read_metadata()
    prefix = "optim_state.optim.state."
    slots = {
        int(key[len(prefix) :].split(".")[0]): {
            "step": torch.empty(value.size, dtype=value.properties.dtype)
        }
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith(prefix) and key.endswith(".step")
    }
    for index in slots:
        for field in ("exp_avg", "exp_avg_sq"):
            if f"{prefix}{index}.{field}" not in metadata.state_dict_metadata:
                raise ValueError("Incomplete saved Adam state")
    dcp.load({"optim_state": {"optim": {"state": slots}}}, checkpoint_id=str(directory))
    return [int(state["step"]) for state in slots.values()]


def inspect_checkpoint(root):
    import torch

    root = Path(root)
    masters = read_group(
        root / "optimizer", "optim_state", "fp32_params", "optim_state.fp32_params."
    )
    model = read_group(root / "model", "model_state", "model", "model_state.model.")
    weights = {name: model["draft_model." + name] for name in masters}
    if any(value.dtype != torch.float32 for value in masters.values()):
        raise ValueError("Saved masters are not FP32")
    for name, master in masters.items():
        if not torch.equal(master.to(weights[name].dtype), weights[name]):
            raise ValueError(f"Actual saved model disagrees with its optimizer master: {name}")
    counts = adam_counts(root / "optimizer")
    if len(counts) != len(masters) or set(counts) != {32}:
        raise ValueError("32-step checkpoint lacks a complete matching Adam frontier")
    return masters, weights, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fast, fast_bf16, counts = inspect_checkpoint(args.fast)
    reference, reference_bf16, ref_counts = inspect_checkpoint(args.reference)
    if fast.keys() != reference.keys():
        raise ValueError("Actual saved trainable parameter identities differ")
    records = {}
    for name in fast:
        records[name] = {
            "fp32": vector_comparison(
                [fast[name].to(args.device)], [reference[name].to(args.device)]
            ),
            "bf16": vector_comparison(
                [fast_bf16[name].to(args.device)], [reference_bf16[name].to(args.device)]
            ),
        }
    worst = sorted(records, key=lambda name: records[name]["fp32"]["relative_l2"], reverse=True)
    result = {
        "step": 32,
        "parameters": len(records),
        "adam_steps": sorted(set(counts + ref_counts)),
        "actual_model_equals_rounded_own_master": True,
        "worst_fp32": [{"name": name, **records[name]["fp32"]} for name in worst[:10]],
        "per_parameter": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_parameter"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
