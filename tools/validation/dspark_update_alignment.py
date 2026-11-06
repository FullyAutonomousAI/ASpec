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

"""Observe native full-chain optimizer updates without changing their arithmetic.

This CP1 diagnosis records 16 update witnesses and full gradients for the first
two updates, followed by all trainable/master/moment tensors after update two.
It is not a product CP restriction or a new training mode.
"""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

# AngelSpec also has a regular tools package; avoid its namespace shadowing.
pipeline = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).with_name("dspark_pipeline_alignment.py")))
)


def cpu_tensor(value):
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().cpu().clone()


def samples(value):
    if value is None:
        return None
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        value = value.full_tensor()
    flat = value.detach().flatten()
    indices = pipeline.gradient_sample_indices(flat.numel(), flat.device)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": flat[indices].float().cpu().tolist(),
    }


def save_new(payload, path):
    import torch

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    torch.save(payload, temporary)
    temporary.rename(path)


def install_update_observer(trainer, run_dir):
    import torch.distributed as dist

    if dist.get_world_size() != 1:
        raise ValueError(
            "This optimizer diagnosis requires CP1; training supports other CP degrees"
        )
    optimizer = trainer.optimizer
    named = [
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    ]
    if len(named) != len(optimizer.model_params) or any(
        parameter is not expected
        for (_, parameter), expected in zip(named, optimizer.model_params, strict=True)
    ):
        raise ValueError("Model/master parameter ordering differs")
    root = Path(run_dir)
    original = optimizer.step

    def step(*args, **kwargs):
        number = trainer.global_step + 1
        before = {}
        gradients = {}
        for (name, parameter), master in zip(named, optimizer.fp32_params, strict=True):
            if parameter.grad is None:
                raise ValueError(f"Missing gradient: {name}")
            before[name] = {
                "model": samples(parameter),
                "master": samples(master),
                "gradient": samples(parameter.grad),
                "placements": [str(p) for p in getattr(parameter, "placements", ())],
            }
            if number <= 2:
                gradients[name] = cpu_tensor(parameter.grad)
        learning_rate = optimizer.get_learning_rate()
        scheduler_before = optimizer.scheduler.last_epoch
        if number <= 2:
            save_new(gradients, root / f"gradients-{number:04d}.pt")
        del gradients
        result = original(*args, **kwargs)
        after = {}
        full = {}
        for (name, parameter), master in zip(named, optimizer.fp32_params, strict=True):
            state = optimizer.optimizer.state[master]
            after[name] = {
                "model": samples(parameter),
                "master": samples(master),
                "exp_avg": samples(state["exp_avg"]),
                "exp_avg_sq": samples(state["exp_avg_sq"]),
                "optimizer_step": state["step"].item(),
            }
            if number == 2:
                full[name] = {
                    "model": cpu_tensor(parameter),
                    "master": cpu_tensor(master),
                    "exp_avg": cpu_tensor(state["exp_avg"]),
                    "exp_avg_sq": cpu_tensor(state["exp_avg_sq"]),
                    "optimizer_step": state["step"].item(),
                }
        if number == 2:
            save_new(full, root / "state-after-0002.pt")
        record = {
            "step": number,
            "learning_rate_used": learning_rate,
            "learning_rate_next": optimizer.get_learning_rate(),
            "scheduler_epoch_before": scheduler_before,
            "scheduler_epoch_after": optimizer.scheduler.last_epoch,
            "reported_global_norm": result.item(),
            "max_grad_norm": optimizer.max_grad_norm,
            "optimizer_defaults": optimizer.optimizer.defaults,
            "before": before,
            "after": after,
        }
        with (root / "update-witnesses.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"UPDATE_AUDIT step={number} lr={learning_rate} norm={result.item()}", flush=True)
        return result

    optimizer.step = step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("angel", "fast"), required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    namespace = "angelspec" if args.role == "angel" else "aspec"
    entry = pipeline.importlib.import_module(namespace + ".train_entry")
    base = entry.TrainerActor

    class AuditActor(base):
        def init(self, *positional, **kwargs):
            result = super().init(*positional, **kwargs)
            install_update_observer(self._trainer, args.run_dir)
            return result

    entry.TrainerActor = AuditActor
    protocol = SimpleNamespace(
        role=args.role,
        fixture=args.fixture,
        run_dir=args.run_dir,
        steps=16,
        cp=1,
        learning_rate=None,
        capture_trajectory=None,
        replay_trajectory=None,
    )
    pipeline.run(protocol)


if __name__ == "__main__":
    main()
