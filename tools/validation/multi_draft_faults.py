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

"""Supervised fault/slow-consumer injection; never used for performance claims."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ray

from aspec.controller.shared_source import SharedSourceState
from aspec.training.trainer_actor import TrainerActor


class FaultSourceState(SharedSourceState):
    def get_prompts(self, count):
        if self._published >= 8:
            self.set_inference_error("injected shared producer failure")
            raise RuntimeError("injected shared producer failure")
        return super().get_prompts(count)


class FaultTrainerActor(TrainerActor):
    def train_from_queue(self, step, num_batches):
        if Path(self.args.output_dir).name == os.environ["ASPEC_FAULT_EXPERIMENT"]:
            action = os.environ["ASPEC_FAULT_ACTION"]
            if step == 1 and action == "cancel_blocked":
                fetcher = self._trainer.data_fetcher
                fetcher = getattr(fetcher, "inner", fetcher)
                observer = fetcher._dataloader.dataset._lifecycle_observer
                ray.get(observer.hub.request_cancel.remote(observer.name, "cancel during blocked step"))
                time.sleep(600)  # The driver must fence us, not wait for this call.
            if action in {"slow", "slow_cancel"}:
                time.sleep(2)
            if (step == 1 and action in {"fail", "cancel"}) or (
                step == 4 and action == "slow_cancel"
            ):
                if action in {"cancel", "slow_cancel"}:
                    fetcher = self._trainer.data_fetcher
                    fetcher = getattr(fetcher, "inner", fetcher)
                    observer = fetcher._dataloader.dataset._lifecycle_observer
                    ray.get(
                        observer.hub.request_cancel.remote(observer.name, "validation cancellation")
                    )
                raise RuntimeError(f"injected independent experiment {action}")
        return super().train_from_queue(step, num_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("fail", "cancel", "cancel_blocked", "slow", "slow_cancel", "source_fail"), required=True
    )
    parser.add_argument("--experiment", default="draft_b")
    args, unknown = parser.parse_known_args()
    from aspec import multi_draft, train_entry

    original = multi_draft.compile_multi_draft_plan

    def compile_test(config):
        for experiment in config.experiments:
            experiment.training.train_env_vars = json.dumps(
                {
                    "ASPEC_FAULT_EXPERIMENT": args.experiment,
                    "ASPEC_FAULT_ACTION": args.action,
                }
            )
        return original(config)

    multi_draft.compile_multi_draft_plan = compile_test
    multi_draft.TrainerActor = FaultTrainerActor
    if args.action == "source_fail":
        multi_draft.SharedSourceController = ray.remote(max_concurrency=32)(FaultSourceState)
    sys.argv = [sys.argv[0], *unknown]
    train_entry.main()


if __name__ == "__main__":
    main()
