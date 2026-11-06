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

"""Compare native SM90 updates at 16 shared weight/optimizer/RNG states.

This interleaves the old and candidate paths on the same resident feature feed.
The next shared state follows the reference trajectory. Diagnostic replay work
is intentionally duplicated and its wall time is not a throughput measurement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path


def differences(reference, actual):
    import torch

    ref_squared = torch.zeros((), device="cuda", dtype=torch.float64)
    delta_squared = torch.zeros_like(ref_squared)
    largest = torch.zeros((), device="cuda")
    for ref, got in zip(reference, actual, strict=True):
        ref, got = ref.detach().float(), got.detach().float()
        delta = got - ref
        ref_squared += ref.double().square().sum()
        delta_squared += delta.double().square().sum()
        largest = torch.maximum(largest, delta.abs().max())
    return {"max_abs": largest.item(),
            "rel_l2": math.sqrt(delta_squared.item()) / max(math.sqrt(ref_squared.item()), 1e-30)}


def snapshot(trainer):
    import torch

    optimizer = trainer.optimizer
    return {
        "model": [p.detach().clone() for p in optimizer.model_params],
        "master": [p.detach().clone() for p in optimizer.fp32_params],
        "optimizer": copy.deepcopy(optimizer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(optimizer.scheduler.state_dict()),
        "global_step": trainer.global_step,
        "cursor": trainer.data_fetcher.cursor,
        "committed": dict(trainer._committed_work_by_step),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def restore(trainer, state):
    import torch

    optimizer = trainer.optimizer
    with torch.no_grad():
        torch._foreach_copy_(optimizer.model_params, state["model"])
        torch._foreach_copy_(optimizer.fp32_params, state["master"])
        for parameter in (*optimizer.model_params, *optimizer.fp32_params):
            parameter.grad = None
    # load_state_dict can retain state tensors, so the frozen witness must not
    # become the next candidate's mutable Adam moments.
    optimizer.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    optimizer.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    trainer.global_step = state["global_step"]
    trainer.data_fetcher.cursor = state["cursor"]
    trainer._committed_work_by_step = dict(state["committed"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--alignment-result", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.alignment_result.exists():
        raise FileExistsError(args.alignment_result)

    import torch
    import torch.distributed as dist

    from aspec.models import dspark as model_module
    from aspec.training.dspark_trainer import DSparkTrainer
    from tools.performance import torchspec_benchmark

    choose = model_module.use_sm90_tiled_l1
    original = DSparkTrainer.train_from_queue
    records = []

    def replay(self, *positional, **kwargs):
        if dist.get_world_size() != 1 or torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("this diagnostic replay requires one SM90 Draft rank")
        if self._pending_train_metrics is not None:
            raise RuntimeError("common-state replay expects immediate native metrics")
        before = snapshot(self)
        optimizer_step = self.optimizer.step
        original_fuse_clip = self.optimizer._fuse_clip_scaling
        gradients = None
        gradient_errors = None
        reference_pass = True

        def observed_update(*values, **named):
            nonlocal gradients, gradient_errors
            current = [p.grad for p in self.optimizer.model_params]
            if any(g is None for g in current):
                raise RuntimeError("missing gradient in full Draft update")
            if reference_pass:
                gradients = [g.detach().clone() for g in current]
            else:
                gradient_errors = differences(gradients, current)
            return optimizer_step(*values, **named)

        self.optimizer.step = observed_update
        try:
            model_module.use_sm90_tiled_l1 = lambda *_: False
            self.optimizer._fuse_clip_scaling = False
            reference_metrics = original(self, *positional, **kwargs)
            after_reference = snapshot(self)
            restore(self, before)
            model_module.use_sm90_tiled_l1 = choose
            self.optimizer._fuse_clip_scaling = original_fuse_clip
            reference_pass = False
            candidate_metrics = original(self, *positional, **kwargs)
            if self.model.l1_loss_backend != "tiled_fp32":
                raise RuntimeError("candidate tiled L1 did not execute")
            optimizer = self.optimizer
            state = optimizer.optimizer.state
            old_states = list(after_reference["optimizer"]["state"].values())
            record = {
                "step": self.global_step,
                "reference_metrics": reference_metrics,
                "candidate_metrics": candidate_metrics,
                "gradients": gradient_errors,
                "model": differences(after_reference["model"], optimizer.model_params),
                "master": differences(after_reference["master"], optimizer.fp32_params),
                "master_updates": differences(
                    (after - prior for after, prior in
                     zip(after_reference["master"], before["master"], strict=True)),
                    (after - prior for after, prior in
                     zip(optimizer.fp32_params, before["master"], strict=True)),
                ),
            }
            for name in ("exp_avg", "exp_avg_sq"):
                record[name] = differences([s[name] for s in old_states],
                                           [state[p][name] for p in optimizer.fp32_params])
            for field in ("train/avg_loss", "train/grad_norm", "train/l1_loss"):
                left, right = reference_metrics[field], candidate_metrics[field]
                record[field] = {"abs_diff": abs(right - left),
                                 "rel_diff": abs(right - left) / max(abs(left), 1e-30)}
            records.append(record)
            args.alignment_result.write_text(json.dumps({
                "protocol": "native 16+ shared model/master/moment/RNG states; reference trajectory",
                "records": records,
                "source_sha256": {
                    suffix: hashlib.sha256(Path(sys.modules["aspec." + suffix].__file__).read_bytes()).hexdigest()
                    for suffix in ("models.dspark", "models.ops.loss", "models.ops.tiled_l1",
                                   "training.optimizer", "training.dspark_trainer")
                },
                "scope": "correctness replay; duplicated work and diagnostic timing",
            }, indent=2) + "\n")
            print("ALIGNMENT " + json.dumps({k: record[k] for k in
                  ("step", "train/avg_loss", "train/grad_norm", "gradients", "master")}), flush=True)
            restore(self, after_reference)
            return candidate_metrics
        finally:
            self.optimizer.step = optimizer_step
            self.optimizer._fuse_clip_scaling = original_fuse_clip
            model_module.use_sm90_tiled_l1 = choose

    DSparkTrainer.train_from_queue = replay
    sys.argv = [sys.argv[0], *remaining]
    try:
        torchspec_benchmark.main()
        if len(records) < 16:
            raise RuntimeError("alignment requires at least 16 shared update states")
    finally:
        DSparkTrainer.train_from_queue = original


if __name__ == "__main__":
    main()
