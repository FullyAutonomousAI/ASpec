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

"""One-time real Fast state transplant into otherwise native Angel training.

This diagnoses numerical-perturbation amplification, not alignment acceptance.
After two normal Angel steps, copy the captured Fast trainable/master/moment
state exactly once. All subsequent arithmetic remains native AngelSpec.
"""

import argparse
import importlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace


def install_transplant(trainer, state_file, run_dir):
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, distribute_tensor

    if dist.get_world_size() != 1:
        raise ValueError("This perturbation diagnosis uses the CP1 Angel reference")
    pipeline = runpy.run_path(str(Path(__file__).with_name("dspark_pipeline_alignment.py")))
    original = trainer._train_step
    done = False

    def train_step(*args, **kwargs):
        nonlocal done
        if trainer.global_step == 2 and not done:
            source = torch.load(state_file, mmap=True, weights_only=True)
            optimizer = trainer.optimizer
            named = [
                (name, parameter)
                for name, parameter in trainer.model.named_parameters()
                if parameter.requires_grad
            ]
            if set(source) != {name for name, _ in named}:
                raise ValueError("Transplant parameter set mismatch")
            for (name, parameter), master in zip(named, optimizer.fp32_params, strict=True):
                targets = {
                    "model": parameter,
                    "master": master,
                    "exp_avg": optimizer.optimizer.state[master]["exp_avg"],
                    "exp_avg_sq": optimizer.optimizer.state[master]["exp_avg_sq"],
                }
                if (
                    optimizer.optimizer.state[master]["step"].item()
                    != source[name]["optimizer_step"]
                ):
                    raise ValueError("Adam step phase mismatch")
                for field, target in targets.items():
                    value = source[name][field].to(target.device)
                    with torch.no_grad():
                        if isinstance(target, DTensor):
                            value = distribute_tensor(value, target.device_mesh, target.placements)
                        target.copy_(value)
                    actual = target.to_local() if isinstance(target, DTensor) else target
                    expected = value.to_local() if isinstance(value, DTensor) else value
                    if not torch.equal(actual, expected):
                        raise ValueError(f"Incomplete transplant: {name}/{field}")
            done = True
            (Path(run_dir) / "transplant.json").write_text(
                json.dumps(
                    {
                        "validation_only": True,
                        "transplant_before_step": 3,
                        "source": str(state_file),
                        "source_sha256": pipeline["file_sha"](state_file),
                        "all_trainable_master_and_moment_tensors_exactly_matched": True,
                        "learning_rate_and_precision_changed": False,
                        "establishes_independent_alignment": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            print("STATE_TRANSPLANT complete before step 3", flush=True)
        return original(*args, **kwargs)

    trainer._train_step = train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    entry = importlib.import_module("angelspec.train_entry")
    base = entry.TrainerActor

    class TransplantActor(base):
        def init(self, *positional, **kwargs):
            result = super().init(*positional, **kwargs)
            install_transplant(self._trainer, args.state_file, args.run_dir)
            return result

    entry.TrainerActor = TransplantActor
    pipeline = runpy.run_path(str(Path(__file__).with_name("dspark_pipeline_alignment.py")))
    pipeline["run"](
        SimpleNamespace(
            role="angel",
            fixture=args.fixture,
            run_dir=args.run_dir,
            steps=16,
            cp=1,
            learning_rate=None,
            capture_trajectory=None,
            replay_trajectory=None,
        )
    )


if __name__ == "__main__":
    main()
