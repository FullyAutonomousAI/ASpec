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

"""Inject a bounded trainer exception/slow reader in a task-owned recipe run."""

import argparse
import json
import sys
import time
from pathlib import Path

from tools.validation.recipe_gpu_smoke import AuditedTrainerActor
from tools.validation.recipe_gpu_smoke import main as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("fail", "slow"), required=True)
    parser.add_argument("--shared", action="store_true")
    args, remaining = parser.parse_known_args()
    action, shared = args.action, args.shared

    class InjectedTrainer(AuditedTrainerActor):
        def train_from_queue(self, step, num_batches):
            selected = self._rank == 0 and (
                not shared or Path(self.args.output_dir).name == "draft_b"
            )
            if selected and step == 1:
                (Path(self.args.output_dir) / "recipe-fault.json").write_text(
                    json.dumps({"action": action, "step": step, "rank": self._rank})
                )
                if action == "fail":
                    raise RuntimeError("Recipe acceptance: injected trainer exception")
                time.sleep(5)
            return super().train_from_queue(step, num_batches)

    sys.argv = [sys.argv[0], *remaining]
    run(trainer_class=InjectedTrainer)


if __name__ == "__main__":
    main()
