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

"""Replay real normalized gradients through both production BF16Optimizers.

Run with torchrun --standalone --nproc-per-node=1. Parameters are streamed one
at a time, retaining world-one DTensors and the measured whole-model clipping
coefficient. AdamW is coordinatewise: this isolates update arithmetic without
rerunning forward/backward or allocating four complete draft models.
"""

import argparse
import importlib
import json
import math
from pathlib import Path


def add_difference(accumulator, reference, candidate):
    import torch

    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    accumulator["reference_squared"] += torch.sum(reference.double().square()).item()
    accumulator["difference_squared"] += torch.sum(difference.double().square()).item()
    accumulator["max_absolute"] = max(accumulator["max_absolute"], difference.abs().max().item())
    accumulator["different_coordinates"] += torch.count_nonzero(difference).item()
    accumulator["coordinates"] += reference.numel()


def new_difference():
    return dict(
        reference_squared=0.0,
        difference_squared=0.0,
        max_absolute=0.0,
        different_coordinates=0,
        coordinates=0,
    )


def finish_difference(value):
    return {
        **value,
        "relative_l2": math.sqrt(value["difference_squared"])
        / max(math.sqrt(value["reference_squared"]), 1e-30),
        "different_fraction": value["different_coordinates"] / max(value["coordinates"], 1),
    }


def replay_parameter(optimizer_class, initial, gradients, witnesses, mesh):
    import torch
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(distribute_tensor(initial.cuda(), mesh, [Shard(0)]))
    )
    optimizer = optimizer_class(
        model,
        lr=6e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        total_steps=100,
        warmup_ratio=0.04,
        min_lr=6e-5,
    )
    original_clip = torch.nn.utils.clip_grad_norm_
    current = None

    def measured_global_clip(parameters, max_norm, *args, **kwargs):
        norm = torch.tensor(current["reported_global_norm"], device="cuda", dtype=torch.float32)
        coefficient = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
        return distribute_tensor(norm, mesh, [Replicate()])

    torch.nn.utils.clip_grad_norm_ = measured_global_clip
    try:
        for gradient, witness in zip(gradients, witnesses, strict=True):
            current = witness
            if optimizer.get_learning_rate() != witness["learning_rate_used"]:
                raise ValueError("Replay learning-rate phase mismatch")
            model.weight.grad = distribute_tensor(gradient.cuda(), mesh, [Shard(0)])
            optimizer.step()
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    master = optimizer.fp32_params[0]
    state = optimizer.optimizer.state[master]
    if state["step"].item() != 2:
        raise ValueError("Replay did not advance exactly two Adam steps")
    return {
        "model": model.weight.to_local().detach(),
        "master": master.to_local().detach(),
        "exp_avg": state["exp_avg"].to_local().detach(),
        "exp_avg_sq": state["exp_avg_sq"].to_local().detach(),
    }


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    mesh = init_device_mesh("cuda", (1,))
    root = Path(args.root)
    roles = ("angel", "fast-cp1")
    witnesses = {
        role: [
            json.loads(line)
            for line in (root / role / "update-witnesses.jsonl").read_text().splitlines()
        ]
        for role in roles
    }
    if any([row["step"] for row in rows] != list(range(1, 17)) for rows in witnesses.values()):
        raise ValueError("Incomplete 16-step native training audit")
    initial = torch.load(Path(args.fixture) / "initial-state.pt", mmap=True, weights_only=True)
    gradients = {
        role: [
            torch.load(root / role / f"gradients-{step:04d}.pt", mmap=True, weights_only=True)
            for step in (1, 2)
        ]
        for role in roles
    }
    actual = {
        role: torch.load(root / role / "state-after-0002.pt", mmap=True, weights_only=True)
        for role in roles
    }
    # Each repository installs the same CuTe compatibility module. Its missing
    # spec only affects importing both repositories in one validation process.
    import sys

    fast_class = importlib.import_module("aspec.training.optimizer").BF16Optimizer
    shim = sys.modules.get("cutlass.utils.ampere_helpers")
    if shim is not None and shim.__spec__ is None:
        shim.__spec__ = importlib.machinery.ModuleSpec("cutlass.utils.ampere_helpers", loader=None)

    classes = {
        "fast": fast_class,
        "angel": importlib.import_module("angelspec.training.optimizer").BF16Optimizer,
    }
    fields = ("model", "master", "exp_avg", "exp_avg_sq")
    sums = {}
    per_parameter = {}

    def compare(label, reference, candidate, name, initial_parameter):
        totals = sums.setdefault(label, {field: new_difference() for field in (*fields, "update")})
        parameter_record = {}
        for field in fields:
            add_difference(totals[field], reference[field], candidate[field])
            if field == "master":
                local = new_difference()
                add_difference(
                    local,
                    reference[field] - initial_parameter,
                    candidate[field] - initial_parameter,
                )
                parameter_record = finish_difference(local)
        add_difference(
            totals["update"],
            reference["master"] - initial_parameter,
            candidate["master"] - initial_parameter,
        )
        per_parameter.setdefault(label, {})[name] = parameter_record

    gradient_sums = {str(step): new_difference() for step in (1, 2)}
    sign_flips = {str(step): 0 for step in (1, 2)}
    names = list(actual["angel"])
    if set(names) != set(actual["fast-cp1"]) or any(
        set(g) != set(names) for rows in gradients.values() for g in rows
    ):
        raise ValueError("Full gradient/state parameter sets differ")
    for index, name in enumerate(names):
        parameter = initial[name]
        initial_parameter = parameter.float().cuda()
        actual_gpu = {
            role: {field: actual[role][name][field].cuda() for field in fields} for role in roles
        }
        replays = {}
        for role in roles:
            for algorithm, optimizer_class in classes.items():
                replays[(algorithm, role)] = replay_parameter(
                    optimizer_class,
                    parameter,
                    [g[name] for g in gradients[role]],
                    witnesses[role][:2],
                    mesh,
                )
            compare(
                "actual_vs_replay_" + role,
                actual_gpu[role],
                replays[("angel" if role == "angel" else "fast", role)],
                name,
                initial_parameter,
            )
            compare(
                "same_gradient_algorithm_" + role,
                replays[("angel", role)],
                replays[("fast", role)],
                name,
                initial_parameter,
            )
        for algorithm in classes:
            compare(
                "gradient_effect_" + algorithm,
                replays[(algorithm, "angel")],
                replays[(algorithm, "fast-cp1")],
                name,
                initial_parameter,
            )
        compare(
            "actual_framework_difference",
            actual_gpu["angel"],
            actual_gpu["fast-cp1"],
            name,
            initial_parameter,
        )
        for step in (1, 2):
            left = gradients["angel"][step - 1][name].float().cuda()
            right = gradients["fast-cp1"][step - 1][name].float().cuda()
            add_difference(gradient_sums[str(step)], left, right)
            sign_flips[str(step)] += torch.count_nonzero(
                (left < 0) & (right > 0) | (left > 0) & (right < 0)
            ).item()
        print(f"REPLAY {index + 1}/{len(names)} {name}", flush=True)
        del replays, actual_gpu, initial_parameter, left, right
    result = {
        "protocol": "real first-two-step gradients / production optimizers / CP1 DTensor replay",
        "source_sha256": {},
        "complete_vector_gradients": {
            step: {**finish_difference(value), "opposite_sign_coordinates": sign_flips[step]}
            for step, value in gradient_sums.items()
        },
        "comparisons": {
            label: {field: finish_difference(value) for field, value in totals.items()}
            for label, totals in sums.items()
        },
        "per_parameter_update_differences": per_parameter,
        "optimizer_defaults": {role: witnesses[role][0]["optimizer_defaults"] for role in roles},
        "assumption": "AdamW coordinatewise; global clip coefficient replayed from each actual run",
    }
    import hashlib

    for algorithm, optimizer_class in classes.items():
        module = importlib.import_module(optimizer_class.__module__)
        result["source_sha256"][algorithm] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
