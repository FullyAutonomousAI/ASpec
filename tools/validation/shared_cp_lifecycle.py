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

"""Resume the public CP recipe, cancel one CP group and fail another.

Diagnostic only: the healthy sibling is deliberately slowed, never benchmarked.
Checkpoint restoration is audited by the existing recipe harness on every rank.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import ray
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from tools.validation import recipe_gpu_smoke


class LifecycleTrainer(recipe_gpu_smoke.AuditedTrainerActor):
    def train_from_queue(self, step, num_batches):
        name = Path(self.args.output_dir).name
        current = self._trainer.global_step
        if current >= 17 and self._rank == 0:
            if name == "eagle3_cp2":
                fetcher = self._trainer.data_fetcher
                fetcher = getattr(fetcher, "inner", fetcher)
                observer = fetcher._dataloader.dataset._lifecycle_observer
                ray.get(observer.hub.request_cancel.remote(name, "CP lifecycle regression"))
                time.sleep(120)  # The driver must fence all CP ranks, not await us.
                raise AssertionError("Cancellation did not fence the CP group")
            if name == "dflash_cp4":
                raise RuntimeError("injected CP4 rank-zero failure")
        if name == "dspark_cp1":
            time.sleep(4)
        result = super().train_from_queue(step, num_batches)
        with (self._audit_root / f"lifecycle-steps{self._audit_suffix}.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self._trainer.global_step, "time": time.time()}) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", required=True)
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, save_snapshot=False)
    experiments = OmegaConf.to_container(config.experiments, resolve=True)
    for experiment in experiments:
        experiment.setdefault("training", {})["load_path"] = str(
            Path(args.resume_from).resolve() / experiment["name"] / "checkpoints" / "iter_0000017"
        )
    sys.argv = [sys.argv[0], "--config", args.config, "--output", args.output, *overrides,
                "experiments=" + json.dumps(experiments)]
    try:
        recipe_gpu_smoke.main(trainer_class=LifecycleTrainer)
    except RuntimeError:
        root = Path(args.output)
        evidence = json.loads((root / "evidence.json").read_text())
        states = {name: row["state"] for name, row in evidence["experiments"].items()}
        assert states == {"dspark_cp1": "completed", "eagle3_cp2": "cancelled", "dflash_cp4": "failed"}, states
        assert evidence["state"] == "completed_with_failed_experiments", evidence["state"]
        assert not evidence["cleanup_errors"], evidence["cleanup_errors"]
        source = evidence["shared_source"]
        assert source["active_experiments"] == [], source
        assert source["pending_source_subscriptions"] == 0, source
        assert evidence["read_owners_fenced"] is True
        for key in ("outstanding_observations", "outstanding_payload_bytes", "pending_production"):
            assert source[key] == 0, (key, source)
        healthy = json.loads((root / "dspark_cp1/evidence.json").read_text())
        assert healthy["health"] == "healthy", healthy["health"]
        print("PASS independent CP cancellation/failure, restored healthy sibling and fenced cleanup")
        return
    raise AssertionError("Expected isolated cancellation/failure did not occur")


if __name__ == "__main__":
    main()
