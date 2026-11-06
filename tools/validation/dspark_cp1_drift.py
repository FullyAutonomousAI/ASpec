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

"""Capture real CP1 inputs and native updates for causal offline diagnostics.

No training arithmetic is replaced. Reuse the unchanged TorchSpec reference,
normal ASpec pipeline and existing full/sampled optimizer audit. The saved
teacher tensors permit kernel/grouping probes without regenerating a teacher.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tools.validation import dspark_torchspec_alignment as alignment
from tools.validation.dspark_update_alignment import cpu_tensor, install_update_observer, save_new


def owned_inputs(kwargs):
    """Copy every actual forward input except the separately stored frozen head."""
    import torch

    result = {}
    for key, value in kwargs.items():
        if key == "lm_head_weight":
            continue
        if isinstance(value, torch.Tensor):
            value = cpu_tensor(value)
        elif isinstance(value, list):
            value = [cpu_tensor(item) for item in value]
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Unsupported forward input: {key}: {type(value)}")
        result[key] = value
    return result


class CapturingReference(alignment.TorchReference):
    def __init__(self, trainer, fixture, run_dir):
        super().__init__(trainer, fixture, run_dir)
        self.input_root = Path(run_dir) / "inputs"
        self.input_root.mkdir(exist_ok=False)
        self.audit_trainer = SimpleNamespace(
            model=self.model, optimizer=self.optimizer, global_step=0
        )
        for name, audited in (("fast-audit", trainer), ("torch-audit", self.audit_trainer)):
            root = Path(run_dir) / name
            root.mkdir(exist_ok=False)
            install_update_observer(audited, root)
        self.microbatch = 0

    def pre_forward(self, module, positional, kwargs):
        super().pre_forward(module, positional, kwargs)
        if self.trainer.global_step == 0 and self.microbatch == 0:
            save_new(cpu_tensor(kwargs["lm_head_weight"]), self.input_root / "lm-head.pt")
        name = f"step-{self.trainer.global_step + 1:04d}-micro-{self.microbatch:02d}.pt"
        save_new(owned_inputs(kwargs), self.input_root / name)
        self.microbatch += 1

    def before_update(self):
        self.audit_trainer.global_step = self.trainer.global_step
        super().before_update()
        self.microbatch = 0


def capture(fixture, run_dir, steps):
    pipeline = alignment.pipeline
    original = pipeline.install_observers

    def install(trainer, role, output, capture_trajectory, replay_trajectory):
        original(trainer, role, output, capture_trajectory, replay_trajectory)
        reference = CapturingReference(trainer, fixture, output)
        trainer.model.register_forward_pre_hook(reference.pre_forward, with_kwargs=True)
        native_step = trainer.optimizer.step

        def step(*positional, **kwargs):
            reference.before_update()
            return native_step(*positional, **kwargs)

        trainer.optimizer.step = step

    pipeline.install_observers = install
    pipeline.run(
        SimpleNamespace(
            role="fast",
            cp=1,
            fixture=fixture,
            run_dir=run_dir,
            steps=steps,
            learning_rate=None,
            capture_trajectory=None,
            replay_trajectory=None,
        )
    )
    alignment.compare(run_dir, steps)
    root = Path(run_dir)
    inputs = sorted((root / "inputs").glob("step-*.pt"))
    if len(inputs) != steps * 2:
        raise ValueError("Capture must contain every real packed microbatch")
    (root / "capture-protocol.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "cp": 1,
                "packed_microbatches": len(inputs),
                "production_arithmetic_changed": False,
                "upstream_changed_behavior": "Exact doc-local anchor replay only",
                "full_gradient_steps": [1, 2],
                "full_state_after_step": 2,
                "sampled_update_steps": list(range(1, steps + 1)),
                "harness_sha256": pipeline.file_sha(__file__),
                "fixture_manifest": json.loads((Path(fixture) / "manifest.json").read_text()),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 16:
        parser.error("A trajectory diagnosis requires at least sixteen steps")
    capture(args.fixture, args.run_dir, args.steps)


if __name__ == "__main__":
    main()
