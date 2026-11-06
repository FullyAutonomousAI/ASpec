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

"""Observe CP's own updates and replay a saved divergent state at another CP.

Native forward/backward, clipping, optimizer, precision and LR are unchanged.
The analytical AdamW oracle runs on detached copies after the real update.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from aspec.commands._checkpoint_audit import add_statistics as add_statistics
from aspec.commands._checkpoint_audit import finish_statistics as finish_statistics
from tools.validation.dspark_cp1_probe import build_model, native_window, wrap_native_fsdp
from tools.validation.dspark_cp_self_replay import owner_inputs
from tools.validation.dspark_torchspec_alignment import pipeline
from tools.validation.dspark_update_alignment import cpu_tensor, save_new


def local_tensor(value):
    from torch.distributed.tensor import DTensor

    return value.to_local() if isinstance(value, DTensor) else value


def adamw_oracle(master, first, second, gradient, step, group):
    """Independent FP64 formula, not an alternative production optimizer."""
    beta1, beta2 = group["betas"]
    master, first, second, gradient = [
        value.double() for value in (master, first, second, gradient)
    ]
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * gradient.square()
    update = (first / (1 - beta1**step)) / ((second / (1 - beta2**step)).sqrt() + group["eps"])
    master = master * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * update
    return {"master": master, "exp_avg": first, "exp_avg_sq": second}


def named_trainables(model, optimizer):
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if len(named) != len(optimizer.model_params) or any(
        parameter is not expected
        for (_, parameter), expected in zip(named, optimizer.model_params, strict=True)
    ):
        raise ValueError("Model/master parameter ordering differs")
    return named


def save_snapshot(model, optimizer, destination, step, cp, with_gradients=False):
    """All ranks participate in each full_tensor; only rank zero retains copies."""
    import torch.distributed as dist

    rank = dist.get_rank()
    parameters = {}
    for (name, parameter), master in zip(
        named_trainables(model, optimizer), optimizer.fp32_params, strict=True
    ):
        state = optimizer.optimizer.state[master]
        values = {
            "model": parameter,
            "master": master,
            "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
        }
        if with_gradients:
            values["gradient"] = parameter.grad
        record = {}
        for field, value in values.items():
            if value is None:
                raise ValueError(f"Missing snapshot tensor: {name}/{field}")
            full = cpu_tensor(value)
            if rank == 0:
                record[field] = full
            del full
        if rank == 0:
            record["optimizer_step"] = float(state["step"])
            parameters[name] = record
    if rank == 0:
        payload = {
            "parameters": parameters,
            "before_step": step,
            "source_cp": cp,
            "learning_rate": optimizer.get_learning_rate(),
            "scheduler": optimizer.scheduler.state_dict(),
            "max_grad_norm": optimizer.max_grad_norm,
            "optimizer_group": {
                key: value
                for key, value in optimizer.optimizer.param_groups[0].items()
                if key != "params"
            },
            "harness_sha256": pipeline.file_sha(__file__),
        }
        save_new(payload, destination)
        print(f"SNAPSHOT {destination}", flush=True)
    dist.barrier()


class NativeUpdateAudit:
    """Full-coordinate local-shard observation, followed by exact global sums."""

    def __init__(self, model, optimizer, root, cp, snapshot_steps):
        self.model, self.optimizer = model, optimizer
        self.root, self.cp, self.snapshot_steps = Path(root), cp, set(snapshot_steps)
        self.named = named_trainables(model, optimizer)

    def before(self, step):
        import torch
        import torch.distributed as dist

        self.lr = self.optimizer.get_learning_rate()
        self.epoch = self.optimizer.scheduler.last_epoch
        self.previous = []
        square = torch.zeros((), device="cuda", dtype=torch.float64)
        for (name, parameter), master in zip(self.named, self.optimizer.fp32_params, strict=True):
            state = self.optimizer.optimizer.state[master]
            if parameter.grad is None:
                raise ValueError(f"Missing gradient: {name}")
            values = {
                "master": local_tensor(master).detach().clone(),
                "gradient": local_tensor(parameter.grad).detach().clone(),
            }
            for field in ("exp_avg", "exp_avg_sq"):
                values[field] = (
                    local_tensor(state[field]).detach().clone()
                    if field in state
                    else torch.zeros_like(values["master"])
                )
            values["step"] = float(state["step"]) if "step" in state else 0.0
            if values["step"] != step - 1:
                raise ValueError(f"Adam step discontinuity: {name}")
            for chunk in values["gradient"].flatten().split(1_000_000):
                square.add_(chunk.double().square().sum())
            self.previous.append(values)
        dist.all_reduce(square)
        self.gradient_norm_fp64 = float(square.sqrt())
        if step in self.snapshot_steps:
            save_snapshot(
                self.model, self.optimizer, self.root / f"before-{step:04d}.pt", step, self.cp, True
            )

    def after(self, step, reported_norm):
        import torch
        import torch.distributed as dist

        fields = ("clipped_gradient", "exp_avg", "exp_avg_sq", "master", "update", "model_copy")
        totals = torch.zeros(len(fields), 5, device="cuda", dtype=torch.float64)
        group = dict(self.optimizer.optimizer.param_groups[0], lr=self.lr)
        coefficient = torch.clamp(
            torch.tensor(self.optimizer.max_grad_norm, device="cuda", dtype=torch.float32)
            / (torch.tensor(reported_norm, device="cuda", dtype=torch.float32) + 1e-6),
            max=1.0,
        )
        for (name, parameter), master, clipped, old in zip(
            self.named,
            self.optimizer.fp32_params,
            self.optimizer.fp32_grads,
            self.previous,
            strict=True,
        ):
            state = self.optimizer.optimizer.state[master]
            if float(state["step"]) != step or parameter.grad is not None:
                raise ValueError(f"Native update/gradient cleanup discontinuity: {name}")
            actual = {
                "master": local_tensor(master).detach(),
                "exp_avg": local_tensor(state["exp_avg"]).detach(),
                "exp_avg_sq": local_tensor(state["exp_avg_sq"]).detach(),
                "model_copy": local_tensor(parameter).detach(),
                "clipped_gradient": local_tensor(clipped).detach(),
            }
            for start in range(0, old["master"].numel(), 1_000_000):
                section = slice(start, start + 1_000_000)
                previous = {
                    key: value.flatten()[section] for key, value in old.items() if key != "step"
                }
                observed = {key: value.flatten()[section] for key, value in actual.items()}
                expected_clip = previous["gradient"].float() * coefficient
                expected = adamw_oracle(
                    previous["master"],
                    previous["exp_avg"],
                    previous["exp_avg_sq"],
                    observed["clipped_gradient"],
                    step,
                    group,
                )
                expected.update(
                    clipped_gradient=expected_clip,
                    model_copy=observed["master"].to(parameter.dtype),
                    update=expected["master"] - previous["master"].double(),
                )
                observed["update"] = observed["master"].double() - previous["master"].double()
                for index, field in enumerate(fields):
                    add_statistics(totals[index], expected[field], observed[field])
        maximum = totals[:, 2].clone()
        totals[:, 2].zero_()
        dist.all_reduce(totals)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        totals[:, 2] = maximum
        record = {
            "step": step,
            "cp": self.cp,
            "learning_rate_used": self.lr,
            "learning_rate_next": self.optimizer.get_learning_rate(),
            "scheduler_epoch_before": self.epoch,
            "scheduler_epoch_after": self.optimizer.scheduler.last_epoch,
            "gradient_norm_fp64": self.gradient_norm_fp64,
            "reported_global_norm": reported_norm,
            "clip_coefficient": float(coefficient),
            "statistics": {
                field: finish_statistics(value)
                for field, value in zip(fields, totals.cpu().tolist(), strict=True)
            },
            "oracle_precision": "FP64 analytical formula on detached real states; native update unchanged",
        }
        if self.optimizer.scheduler.last_epoch != self.epoch + 1:
            raise ValueError("Scheduler did not advance once")
        if dist.get_rank() == 0:
            with (self.root / "full-update-audit.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"FULL_UPDATE_AUDIT step={step} norm={reported_norm} update_rel={record['statistics']['update']['relative_l2']}",
                flush=True,
            )
        self.previous = []
        if step in self.snapshot_steps:
            save_snapshot(
                self.model, self.optimizer, self.root / f"after-{step:04d}.pt", step + 1, self.cp
            )


def load_snapshot(model, optimizer, payload):
    """Restore all real model/master/moments and the original schedule phase."""
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    source = payload["parameters"]
    named = named_trainables(model, optimizer)
    if set(source) != {name for name, _ in named}:
        raise ValueError("Snapshot parameter sets differ")
    with torch.no_grad():
        for (name, parameter), master in zip(named, optimizer.fp32_params, strict=True):
            entry = source[name]
            state = optimizer.optimizer.state[master]
            state.update(
                step=torch.tensor(
                    entry["optimizer_step"], dtype=torch.float32, device=master.device
                ),
                exp_avg=torch.zeros_like(master),
                exp_avg_sq=torch.zeros_like(master),
            )
            for field, target in {
                "model": parameter,
                "master": master,
                "exp_avg": state["exp_avg"],
                "exp_avg_sq": state["exp_avg_sq"],
            }.items():
                value = entry[field].to(target.device)
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError(f"Snapshot shape/dtype differs: {name}/{field}")
                if isinstance(target, DTensor):
                    value = distribute_tensor(value, target.device_mesh, target.placements)
                target.copy_(value)
                if not torch.equal(local_tensor(target), local_tensor(value)):
                    raise ValueError(f"Snapshot copy was not exact: {name}/{field}")
    for key, value in payload["optimizer_group"].items():
        if optimizer.optimizer.param_groups[0][key] != value:
            if key != "lr":
                raise ValueError(f"Optimizer contract changed: {key}")
            optimizer.optimizer.param_groups[0][key] = value
    optimizer.scheduler.load_state_dict(payload["scheduler"])
    if optimizer.get_learning_rate() != payload["learning_rate"]:
        raise ValueError("Restored learning-rate phase differs")


def probe(args):
    import torch
    import torch.distributed as dist

    from aspec.utils.distributed import init_direct_cp_groups

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != args.cp:
        raise ValueError("Probe requires world size equal to CP")
    init_direct_cp_groups(args.cp)
    root = Path(args.output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    model = wrap_native_fsdp(
        build_model(args.fixture, "aspec", "fa4", args.cp), args.fixture, "aspec"
    )
    config = OmegaConf.load(Path(args.fixture) / "fast.yaml").training
    optimizer = importlib.import_module("aspec.training.optimizer").BF16Optimizer(
        model.draft_model,
        lr=config.learning_rate,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        total_steps=config.lr_total_steps,
        warmup_ratio=config.warmup_ratio,
    )
    payload = torch.load(args.state, mmap=True, weights_only=True)
    load_snapshot(model, optimizer, payload)
    step = payload["before_step"]
    del payload
    inputs = Path(args.captured) / "inputs"
    paths = sorted(inputs.glob(f"step-{step:04d}-micro-*.pt"))
    if len(paths) != 2:
        raise ValueError("Both real microbatches required")
    head = torch.load(inputs / "lm-head.pt", weights_only=True).cuda()
    packed = [owner_inputs(torch.load(path, weights_only=True), rank, args.cp) for path in paths]
    local = native_window(model, packed, head, "aspec", False, optimizer)
    metrics = ("loss", "ce", "l1", "confidence")
    totals = torch.tensor(
        [local["denominator"], *[local[key] * local["denominator"] for key in metrics]],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(totals)
    record = {key: float(totals[index + 1] / totals[0]) for index, key in enumerate(metrics)}
    gradients = {}
    for name, parameter in named_trainables(model, optimizer):
        full = cpu_tensor(parameter.grad)
        if rank == 0:
            gradients[name] = full
        del full
    if rank == 0:
        save_new(gradients, root / "gradients.pt")
    del gradients
    audit = NativeUpdateAudit(model, optimizer, root, args.cp, [])
    audit.before(step)
    norm = float(optimizer.step())
    audit.after(step, norm)
    save_snapshot(model, optimizer, root / "after.pt", step + 1, args.cp)
    if rank == 0:
        record.update(
            cp=args.cp,
            step=step,
            denominator=float(totals[0]),
            gradient_norm=norm,
            source_state_sha256=pipeline.file_sha(args.state),
            all_loaded_state_coordinates_bitwise_equal=True,
            input_sha256=[pipeline.file_sha(path) for path in paths],
            semantics_precision_lr_changed=False,
        )
        (root / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fixture", "captured", "state", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--cp", type=int, required=True)
    args = parser.parse_args()
    if args.cp < 1:
        parser.error("CP must be positive")
    probe(args)


if __name__ == "__main__":
    main()
