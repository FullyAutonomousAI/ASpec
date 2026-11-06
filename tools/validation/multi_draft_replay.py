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

"""Capture shared-source training and replay each experiment through its native trainer.

Diagnostic only: CPU capture and whole-state hashes are intentionally excluded
from performance runs. No model, optimizer, loss, precision or LR modifications.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from aspec.commands._checkpoint_audit import compare_checkpoint as compare_checkpoint
from aspec.commands._checkpoint_audit import cpu_values as cpu_values
from aspec.commands._checkpoint_audit import state_hashes as state_hashes
from aspec.commands._checkpoint_audit import tensor_hash as tensor_hash
from aspec.training.trainer_actor import TrainerActor


def training_snapshot(trainer, gradients=None):
    """Trainable full coordinates and runtime state, without modifying training."""
    import torch

    names = [(name, p) for name, p in trainer.draft_model.named_parameters() if p.requires_grad]
    if [id(p) for _, p in names] != [id(p) for p in trainer.optimizer.model_params]:
        raise ValueError("snapshot model/master ordering differs")
    parameters = {}
    for (name, model), master in zip(names, trainer.optimizer.fp32_params, strict=True):
        state = trainer.optimizer.optimizer.state[master]
        parameters[name] = cpu_values(
            {
                "model": model,
                "master": master,
                **{key: state.get(key) for key in ("exp_avg", "exp_avg_sq", "step")},
            }
        )
        if gradients is not None:
            parameters[name]["gradient"] = gradients[name]
    return {
        "parameters": parameters,
        "scheduler": cpu_values(trainer.lr_scheduler.state_dict()),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def install_audit(trainer, destination, *, capture, hash_states=True, tensor_from_step=None):
    import torch

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    (root / "initial.json").write_text(json.dumps(state_hashes(trainer)))
    if tensor_from_step is not None:
        torch.save(training_snapshot(trainer), root / "initial-state.pt")
    records = []
    original_step = trainer.optimizer.step
    original_forward = trainer._forward
    window = []

    def audited_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        numerator, denominator = result[6]
        window.append((float(numerator.detach()), float(denominator.detach())))
        return result

    trainer._forward = audited_forward

    def audited_update(*args, **kwargs):
        step = trainer.global_step + 1
        gradients = None
        if tensor_from_step is not None and step >= tensor_from_step:
            gradients = {
                name: cpu_values(p.grad)
                for name, p in trainer.draft_model.named_parameters()
                if p.requires_grad
            }
        gradient = (
            tensor_hash(
                (name, p.grad)
                for name, p in trainer.draft_model.named_parameters()
                if p.requires_grad
            )
            if hash_states
            else None
        )
        lr = trainer.optimizer.get_learning_rate()
        norm = original_step(*args, **kwargs)
        if gradients is not None:
            torch.save(training_snapshot(trainer, gradients), root / f"state-{step:04d}.pt")
        record = {
            "step": trainer.global_step + 1,
            "learning_rate": lr,
            "loss": sum(value[0] for value in window)
            / max(sum(value[1] for value in window), 1e-12),
            "gradient_norm": float(norm),
            "gradient": gradient,
            "updates": state_hashes(trainer) if hash_states else None,
        }
        records.append(record)
        window.clear()
        with (root / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return norm

    trainer.optimizer.step = audited_update
    if capture:
        original_iter = trainer._iter_batches_from_queue

        def captured_batches(count):
            for index, batch in enumerate(original_iter(count)):
                payload = {
                    key: cpu_values(value)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor) or key == "_aspec_work"
                }
                payload["_audit_rng"] = torch.random.get_rng_state()
                payload["_audit_cuda_rng"] = torch.cuda.get_rng_state()
                torch.save(
                    payload, root / f"step-{trainer.global_step + 1:04d}-micro-{index:02d}.pt"
                )
                yield batch

        trainer._iter_batches_from_queue = captured_batches
    return records


class CaptureTrainerActor(TrainerActor):
    def init(self, args, role, mooncake_config=None, with_ref=False):
        if args.world_size != 1:
            raise ValueError("first replay audit uses one GPU per independent Draft")
        result = super().init(args, role, mooncake_config, with_ref)
        first = os.environ.get("ASPEC_AUDIT_TENSOR_FROM_STEP")
        install_audit(
            self._trainer,
            Path(args.output_dir) / "replay-capture",
            capture=True,
            hash_states=first is None,
            tensor_from_step=int(first) if first else None,
        )
        return result


def replay(args):
    import torch
    import torch.distributed as dist

    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan
    from aspec.train_entry import _get_draft_model_config, _validate_and_configure_dflash
    from aspec.training.dspark_trainer import DSparkTrainer
    from aspec.utils.distributed import init_gloo_group

    experiment = Path(args.experiment)
    captured = experiment / "replay-capture"
    config = load_config(config_path=str(experiment / "config.yaml"), save_snapshot=False)
    config.output_dir = args.output
    config.training.save_interval = 0
    config.training.save_final_checkpoint = False
    runtime = compile_execution_plan(config).to_runtime_args()
    if runtime.world_size != 1:
        raise ValueError("replay requires a world-one Draft")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    init_gloo_group()
    runtime.rank = 0
    draft = _get_draft_model_config(runtime)
    _validate_and_configure_dflash(runtime, draft)
    trainer = DSparkTrainer(runtime)
    trainer.init_model(draft, runtime.target_model_path)
    records = install_audit(trainer, args.output, capture=False, hash_states=not args.numeric)
    reference_initial = json.loads((captured / "initial.json").read_text())
    observed_initial = json.loads((Path(args.output) / "initial.json").read_text())
    if reference_initial != observed_initial:
        raise ValueError("standalone initial model/optimizer differs from captured experiment")
    reference = [json.loads(line) for line in (captured / "updates.jsonl").read_text().splitlines()]
    if len(reference) < 16:
        raise ValueError("at least sixteen complete optimizer steps are required")
    for row in reference:
        paths = sorted(captured.glob(f"step-{row['step']:04d}-micro-*.pt"))
        if len(paths) != runtime.draft_accumulation_steps:
            raise ValueError("incomplete captured accumulation window")

        def batches(_count):
            for path in paths:
                batch = torch.load(path, weights_only=True)
                torch.random.set_rng_state(batch.pop("_audit_rng"))
                torch.cuda.set_rng_state(batch.pop("_audit_cuda_rng"))
                batch = {
                    key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                yield batch

        trainer._iter_batches_from_queue = batches
        result = trainer._train_core_from_queue(row["step"] - 1, len(paths))
        if hasattr(result, "finalize"):
            result = result.finalize()
        if records[-1]["step"] != row["step"]:
            raise ValueError("replay optimizer progress differs")
    comparisons = []
    for ref, observed in zip(reference, records, strict=True):
        norm_abs = abs(ref["gradient_norm"] - observed["gradient_norm"])
        loss_abs = abs(ref["loss"] - observed["loss"])
        exact = (
            ref["gradient"] == observed["gradient"] and ref["updates"] == observed["updates"]
            if not args.numeric
            else None
        )
        comparisons.append(
            {
                "step": ref["step"],
                "whole_state_bitwise_equal": exact,
                "loss_absolute_diff": loss_abs,
                "loss_relative_diff": loss_abs / max(abs(ref["loss"]), 1e-12),
                "gradient_norm_absolute_diff": norm_abs,
                "gradient_norm_relative_diff": norm_abs / max(abs(ref["gradient_norm"]), 1e-12),
            }
        )
    summary = {
        "diagnostic_only": True,
        "steps": len(reference),
        "comparisons": comparisons,
        "loss_metric": "pooled_native_dspark_combined_objective",
        "final_native_metrics": result,
        "all_states_equal": (
            all(row["whole_state_bitwise_equal"] for row in comparisons)
            if not args.numeric
            else None
        ),
        "semantics_precision_lr_changed": False,
    }
    if args.numeric:
        checkpoints = sorted((experiment / "checkpoints").glob("iter_*"))
        directory = checkpoints[-1]
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata["global_step"] != len(reference):
            raise ValueError("checkpoint does not represent the final captured optimizer state")
        summary["final_state_full_coordinate_differences"] = compare_checkpoint(trainer, directory)
    (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    trainer.close()
    dist.destroy_process_group()
    if not args.numeric and not summary["all_states_equal"]:
        raise RuntimeError(
            "independent trajectory differs; inspect per-step hashes before acceptance"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture = subparsers.add_parser("capture", add_help=False)
    capture.add_argument("--second-steps", type=int)
    capture.add_argument("--tensor-from-step", type=int)
    reference = subparsers.add_parser("replay")
    reference.add_argument("--experiment", required=True)
    reference.add_argument("--output", required=True)
    reference.add_argument("--numeric", action="store_true")
    args, unknown = parser.parse_known_args()
    if args.mode == "capture":
        from aspec import multi_draft, train_entry

        multi_draft.TrainerActor = CaptureTrainerActor
        if args.second_steps is not None or args.tensor_from_step is not None:
            original_compile = multi_draft.compile_multi_draft_plan

            def compile_with_horizon(config):
                if args.second_steps is not None:
                    config.experiments[1].training.num_train_steps = args.second_steps
                    config.experiments[1].training.lr_total_steps = args.second_steps
                if args.tensor_from_step is not None:
                    for experiment in config.experiments:
                        environment = json.loads(experiment.training.get("train_env_vars", "{}"))
                        environment["ASPEC_AUDIT_TENSOR_FROM_STEP"] = str(args.tensor_from_step)
                        experiment.training.train_env_vars = json.dumps(environment)
                return original_compile(config)

            multi_draft.compile_multi_draft_plan = compile_with_horizon
        sys.argv = [sys.argv[0], *unknown]
        train_entry.main()
    else:
        if unknown:
            parser.error(f"unexpected replay arguments: {unknown}")
        replay(args)


if __name__ == "__main__":
    main()
