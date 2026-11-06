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

"""Checkpoint audit helpers adapted from the existing ASpec validation tools.

Retains their full-state, scheduler and RNG gates for installed Quickstart.
"""

import hashlib
import math


def cpu_values(value):
    import torch

    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_values(item) for item in value)
    return value


def tensor_hash(tensors):
    import torch

    digest = hashlib.sha256()
    count = 0
    for name, tensor in tensors:
        if tensor is None:
            digest.update((name + ":none").encode())
            continue
        tensor = tensor.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        cpu = tensor.cpu().contiguous()
        digest.update(f"{name}:{cpu.dtype}:{tuple(cpu.shape)}".encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += cpu.numel()
        del cpu
    return {"sha256": digest.hexdigest(), "coordinates": count}


def state_hashes(trainer):
    optimizer = trainer.optimizer
    result = {
        "model": tensor_hash(trainer.draft_model.named_parameters()),
        "master": tensor_hash((str(i), p) for i, p in enumerate(optimizer.fp32_params)),
    }
    for key in ("exp_avg", "exp_avg_sq", "step"):
        result[key] = tensor_hash(
            (str(i), optimizer.optimizer.state[p].get(key))
            for i, p in enumerate(optimizer.fp32_params)
        )
    return result


def compare_checkpoint(trainer, directory):
    """Read original full states into detached CPU buffers; never mutate replay."""
    import torch
    import torch.distributed.checkpoint as dcp

    from aspec.training.checkpoint import ModelState, OptimizerState

    candidate_model = ModelState(trainer.model).state_dict()
    candidate_optimizer = OptimizerState(trainer.model, trainer.optimizer).state_dict()
    original_model = cpu_values(candidate_model)
    original_optimizer = cpu_values(candidate_optimizer)
    dcp.load({"model_state": original_model}, checkpoint_id=str(directory / "model"))
    dcp.load({"optim_state": original_optimizer}, checkpoint_id=str(directory / "optimizer"))
    statistics = {
        key: torch.zeros(5, device="cuda", dtype=torch.float64)
        for key in ("model", "master", "exp_avg", "exp_avg_sq")
    }
    worst_parameters = {key: {"relative_l2": 0, "name": None} for key in statistics}

    def compare(key, name, reference, actual):
        reference = reference.reshape(-1)
        actual = actual.reshape(-1)
        if reference.shape != actual.shape:
            raise ValueError("checkpoint tensor shape differs")
        # Bound temporary GPU memory independently of model/optimizer size.
        local = torch.zeros(5, device="cuda", dtype=torch.float64)
        for start in range(0, actual.numel(), 4 * 1024 * 1024):
            add_statistics(
                local,
                reference[start : start + 4 * 1024 * 1024].cuda(),
                actual[start : start + 4 * 1024 * 1024],
            )
        statistics[key][:2].add_(local[:2])
        statistics[key][2].copy_(torch.maximum(statistics[key][2], local[2]))
        statistics[key][3:].add_(local[3:])
        result = finish_statistics(local.cpu().tolist())
        if result["relative_l2"] > worst_parameters[key]["relative_l2"]:
            worst_parameters[key] = {"name": str(name), **result}

    for name, actual in candidate_model["model"].items():
        compare("model", name, original_model["model"][name], actual)
    for name, actual in candidate_optimizer["fp32_params"].items():
        compare("master", name, original_optimizer["fp32_params"][name], actual)
    for identity, state in candidate_optimizer["optim"]["state"].items():
        original = original_optimizer["optim"]["state"][identity]
        if float(original["step"]) != float(state["step"]):
            raise ValueError("optimizer step differs")
        for key in ("exp_avg", "exp_avg_sq"):
            name = list(candidate_optimizer["fp32_params"])[int(identity)]
            compare(key, name, original[key], state[key])
    return {
        key: {**finish_statistics(value.cpu().tolist()), "worst_parameter": worst_parameters[key]}
        for key, value in statistics.items()
    }


def audit_state(trainer):
    import torch

    return {
        "global_step": trainer.global_step,
        "state": state_hashes(trainer),
        "scheduler": cpu_values(trainer.lr_scheduler.state_dict()),
        "torch_rng_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }


def add_statistics(total, reference, actual):
    import torch

    reference, actual = reference.double(), actual.double()
    delta = actual - reference
    total[0].add_(reference.square().sum())
    total[1].add_(delta.square().sum())
    total[2].copy_(torch.maximum(total[2], delta.abs().max()))
    total[3].add_((reference != actual).sum())
    total[4].add_(reference.numel())


def finish_statistics(value):
    norm, delta, maximum, different, count = value
    return {
        "reference_l2": math.sqrt(norm),
        "difference_l2": math.sqrt(delta),
        "relative_l2": math.sqrt(delta / max(norm, 1e-30)),
        "max_absolute": maximum,
        "different_coordinates": int(different),
        "coordinates": int(count),
    }
