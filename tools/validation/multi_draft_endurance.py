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

"""Native long-run acceptance with bounded slow/recovery/failure injection.

Diagnostic only. Records source residency, independent commits and process RSS;
does not change the native loss, optimizer, learning rate or numerical precision.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import ray

from aspec.controller.shared_source import SharedSourceState
from aspec.training.trainer_actor import TrainerActor


class ObservedSourceState(SharedSourceState):
    def status(self):
        with self._lock:
            result = super().status()
            result["experiments"] = {
                name: {
                    "lifecycle": child.get_work_lifecycle_status(),
                    "ready": child.get_ready_pool_size(),
                    "epoch": self._committed_epoch[name],
                    "tracked_epochs": len(self._epochs[name]),
                    "tracked_descriptors": len(self._descriptors[name]),
                }
                for name, child in self.experiments.items()
            }
            return result


class EnduranceTrainerActor(TrainerActor):
    def train_from_queue(self, step, num_batches):
        if Path(self.args.output_dir).name == os.environ["ASPEC_ENDURANCE_EXPERIMENT"]:
            if step < int(os.environ["ASPEC_ENDURANCE_SLOW_STEPS"]):
                time.sleep(float(os.environ["ASPEC_ENDURANCE_DELAY"]))
            if step == int(os.environ["ASPEC_ENDURANCE_FAIL_STEP"]):
                raise RuntimeError("injected isolated failure after slow-consumer recovery")
        return super().train_from_queue(step, num_batches)


def observe_session(root, stopped):
    destination = root / "endurance-observations.jsonl"
    process = psutil.Process()
    hub = None
    started = time.monotonic()
    while not stopped.wait(2):
        try:
            evidence = json.loads((root / "evidence.json").read_text())
            if hub is None and "control" in evidence:
                hub = ray.get_actor(
                    evidence["control"]["actor_name"], namespace=evidence["control"]["namespace"]
                )
            if hub is None:
                continue
            row = {
                "elapsed_seconds": time.monotonic() - started,
                "source": ray.get(hub.status.remote(), timeout=5),
                "processes": [],
            }
            for child in [process, *process.children(recursive=True)]:
                try:
                    row["processes"].append(
                        {
                            "pid": child.pid,
                            "name": child.name(),
                            "rss_bytes": child.memory_info().rss,
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            with destination.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        except Exception as exc:
            # Diagnostics must not terminate a production run. Preserve gaps.
            if root.exists():
                with destination.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "elapsed_seconds": time.monotonic() - started,
                                "observation_error": str(exc),
                            }
                        )
                        + "\n"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="draft_b")
    parser.add_argument("--slow-steps", type=int, required=True)
    parser.add_argument("--delay-seconds", type=float, required=True)
    parser.add_argument("--fail-step", type=int)
    args, unknown = parser.parse_known_args()
    if (
        args.slow_steps < 0
        or args.delay_seconds < 0
        or (args.fail_step is not None and args.fail_step <= args.slow_steps)
    ):
        parser.error("require nonnegative slow phase and failure after recovery phase")
    from aspec import multi_draft, train_entry

    original_compile = multi_draft.compile_multi_draft_plan
    original_run = multi_draft.run_multi_draft

    def compile_test(config):
        for experiment in config.experiments:
            environment = json.loads(experiment.training.get("train_env_vars", "{}"))
            environment.update(
                {
                    "ASPEC_ENDURANCE_EXPERIMENT": args.experiment,
                    "ASPEC_ENDURANCE_SLOW_STEPS": str(args.slow_steps),
                    "ASPEC_ENDURANCE_DELAY": str(args.delay_seconds),
                    "ASPEC_ENDURANCE_FAIL_STEP": str(
                        args.fail_step if args.fail_step is not None else -1
                    ),
                }
            )
            experiment.training.train_env_vars = json.dumps(environment)
        return original_compile(config)

    def run_observed(runtime, plan):
        stopped = threading.Event()
        monitor = threading.Thread(
            target=observe_session, args=(Path(runtime.output_dir), stopped), daemon=True
        )
        monitor.start()
        try:
            return original_run(runtime, plan)
        finally:
            stopped.set()
            monitor.join(timeout=10)

    multi_draft.compile_multi_draft_plan = compile_test
    multi_draft.run_multi_draft = run_observed
    multi_draft.TrainerActor = EnduranceTrainerActor
    multi_draft.SharedSourceController = ray.remote(max_concurrency=32)(ObservedSourceState)
    sys.argv = [sys.argv[0], *unknown]
    train_entry.main()


if __name__ == "__main__":
    main()
