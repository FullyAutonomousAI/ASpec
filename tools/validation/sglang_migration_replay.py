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

"""Bounded old/new native training replay, not a production or benchmark mode.

Reuse saved real Target features, anchors and initial weights. The reference
publishes one full state/gradient window at a time. The candidate checks both
common-state computation/updates and its independent BF16 trajectory. Native
REPLICATE, trainer normalization, clipping, AdamW and LR are unchanged.
"""

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from aspec.training.fsdp import apply_fsdp2
from aspec.training.optimizer import BF16Optimizer
from tools.validation.dspark_cp1_probe import build_model, native_window
from tools.validation.dspark_cp_self_replay import owner_inputs
from tools.validation.dspark_cp_state_audit import add_statistics, finish_statistics
from tools.validation.dspark_pipeline_alignment import file_sha
from tools.validation.dspark_update_alignment import save_new


def wait_for(path, timeout=600):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Missing replay handoff: {path}")
        time.sleep(0.2)


def named(model, optimizer):
    values = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if [id(p) for _, p in values] != [id(p) for p in optimizer.model_params]:
        raise ValueError("Replay model/master ordering differs")
    return values


def snapshot(model, optimizer):
    parameters = {}
    for (name, p), master in zip(named(model, optimizer), optimizer.fp32_params, strict=True):
        state = optimizer.optimizer.state.get(master, {})
        parameters[name] = {
            "model": p.detach().cpu().clone(),
            "master": master.detach().cpu().clone(),
            **{key: value.detach().cpu().clone() for key, value in state.items()},
        }
    return {
        "parameters": parameters,
        "buffers": {name: b.detach().cpu().clone() for name, b in model.named_buffers()},
        "scheduler": optimizer.scheduler.state_dict(),
        "learning_rate": optimizer.get_learning_rate(),
    }


def restore(model, optimizer, payload):
    names = named(model, optimizer)
    if set(payload["parameters"]) != {name for name, _ in names}:
        raise ValueError("Replay state parameter sets differ")
    with torch.no_grad():
        for (name, p), master in zip(names, optimizer.fp32_params, strict=True):
            source = payload["parameters"][name]
            p.copy_(source["model"])
            master.copy_(source["master"])
            state = optimizer.optimizer.state[master]
            state.clear()
            state.update({key: value.to(master.device) for key, value in source.items()
                          if key not in {"model", "master"}})
        own_buffers = dict(model.named_buffers())
        if own_buffers.keys() != payload["buffers"].keys():
            raise ValueError("Replay state buffer sets differ")
        for name, b in own_buffers.items():
            b.copy_(payload["buffers"][name])
    optimizer.scheduler.load_state_dict(payload["scheduler"])
    optimizer.optimizer.param_groups[0]["lr"] = payload["learning_rate"]


def build(args):
    model = build_model(args.fixture, "aspec", args.backend)
    config = OmegaConf.load(Path(args.fixture) / "fast.yaml").training
    config.fsdp_strategy = "REPLICATE"  # Current product default, both sides.
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
    model = apply_fsdp2(model, mesh=mesh, args=config,
                        modules_to_shard=list(model.draft_model.layers))
    optimizer = BF16Optimizer(
        model, lr=config.learning_rate, min_lr=config.min_lr,
        weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
        total_steps=config.lr_total_steps, warmup_ratio=config.warmup_ratio,
    )
    return model, optimizer


def window(model, optimizer, packed, head):
    optimizer.zero_grad()
    record = native_window(model, packed, head, "aspec", False, optimizer)
    if not all(math.isfinite(v) for v in record.values()):
        raise ValueError("Non-finite replay loss")
    return record


@torch.no_grad()
def gradient_difference(model, optimizer, reference):
    total = torch.zeros(5, dtype=torch.float64, device="cuda")
    for name, p in named(model, optimizer):
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError(f"Missing/non-finite gradient: {name}")
        add_statistics(total, reference[name].cuda(), p.grad)
    return finish_statistics(total.tolist())


@torch.no_grad()
def state_difference(model, optimizer, before, after, previous):
    totals = {key: torch.zeros(5, dtype=torch.float64, device="cuda")
              for key in ("model", "master", "exp_avg", "exp_avg_sq", "update")}
    for index, ((name, p), master) in enumerate(
        zip(named(model, optimizer), optimizer.fp32_params, strict=True)
    ):
        state = optimizer.optimizer.state[master]
        actual = {"model": p, "master": master,
                  "exp_avg": state["exp_avg"], "exp_avg_sq": state["exp_avg_sq"]}
        source = after["parameters"][name]
        for key, value in actual.items():
            add_statistics(totals[key], source[key].cuda(), value)
        add_statistics(totals["update"],
                       source["master"].cuda() - before["parameters"][name]["master"].cuda(),
                       master - previous[index])
    return {key: finish_statistics(value.tolist()) for key, value in totals.items()}


def scalar_differences(actual, reference):
    return {key: {"absolute_diff": abs(actual[key] - value),
                  "relative_diff": abs(actual[key] - value) / max(abs(value), 1e-30)}
            for key, value in reference.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "candidate"), required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--captured", required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--backend", choices=("fa4", "cudnn_flex_attention"), default="fa4")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 1:
        raise ValueError("This migration control requires a single REPLICATE rank")
    root = Path(args.exchange)
    role_dir = root / args.role
    role_dir.mkdir(parents=True, exist_ok=False)
    inputs = Path(args.captured) / "inputs"
    paths = sorted(inputs.glob("step-*-micro-*.pt"))
    if len(paths) != 32:
        raise ValueError("Exactly sixteen saved two-microbatch windows are required")
    head = torch.load(inputs / "lm-head.pt", weights_only=True).cuda()
    identity = {
        "role": args.role, "backend": args.backend, "strategy": "REPLICATE",
        "steps": 16, "harness_sha256": file_sha(__file__),
        "initial_state_sha256": file_sha(Path(args.fixture) / "initial-state.pt"),
        "input_sha256": [file_sha(path) for path in paths],
        "packages": {name: version(name) for name in
                     ("torch", "flash-attn-4", "nvidia-cutlass-dsl", "transformers")},
    }
    (role_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    model, optimizer = build(args)
    if args.role == "reference":
        save_new(snapshot(model, optimizer), root / "state-0000.pt")
        independent = independent_optimizer = None
    else:
        independent, independent_optimizer = build(args)
        wait_for(root / "state-0000.pt")
        initial = torch.load(root / "state-0000.pt", mmap=True, weights_only=True)
        restore(independent, independent_optimizer, initial)
        del initial
    for step in range(1, 17):
        packed = [owner_inputs(torch.load(path, weights_only=True), 0, 1)
                  for path in paths if path.name.startswith(f"step-{step:04d}-")]
        before_path, after_path = root / f"state-{step-1:04d}.pt", root / f"state-{step:04d}.pt"
        grad_path, metrics_path = root / f"grad-{step:04d}.pt", root / f"metrics-{step:04d}.json"
        done_path = root / f"done-{step:04d}"
        if args.role == "reference":
            record = window(model, optimizer, packed, head)
            save_new({name: p.grad.detach().cpu().clone() for name, p in named(model, optimizer)}, grad_path)
            record["gradient_norm"] = float(optimizer.step())
            if not math.isfinite(record["gradient_norm"]):
                raise ValueError("Non-finite reference gradient norm")
            save_new(snapshot(model, optimizer), after_path)
            metrics_path.write_text(json.dumps(record) + "\n")
            with (role_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **record}) + "\n")
            wait_for(done_path)
            # Only this producer's transient, already-compared payloads are
            # removed. Keep the final state, metrics and comparison evidence.
            before_path.unlink()
            grad_path.unlink()
        else:
            wait_for(metrics_path)
            before = torch.load(before_path, mmap=True, weights_only=True)
            after = torch.load(after_path, mmap=True, weights_only=True)
            gradients = torch.load(grad_path, mmap=True, weights_only=True)
            reference = json.loads(metrics_path.read_text())
            comparisons = {}
            for mode, current_model, current_optimizer in (
                ("common_state", model, optimizer),
                ("independent", independent, independent_optimizer),
            ):
                if mode == "common_state":
                    restore(current_model, current_optimizer, before)
                previous = [p.detach().clone() for p in current_optimizer.fp32_params]
                record = window(current_model, current_optimizer, packed, head)
                gradient = gradient_difference(current_model, current_optimizer, gradients)
                record["gradient_norm"] = float(current_optimizer.step())
                comparisons[mode] = {
                    "values": record, "scalars": scalar_differences(record, reference),
                    "gradient": gradient,
                    "state": state_difference(current_model, current_optimizer, before, after, previous),
                }
                del previous
            result = {"step": step, **comparisons}
            with (role_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            del before, after, gradients
            done_path.touch(exist_ok=False)
        if args.role == "reference":
            print(f"Reference window {step}/16 compared and retired", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
