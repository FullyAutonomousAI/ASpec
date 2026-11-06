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

"""Audit externally injected whole-node loss, then exact recovery on an owned cluster.

The fault controller is external: stop the selected task-owned worker node or use
infrastructure controls only after node-loss-armed.json appears. This program
never shuts down a machine or another task's cluster.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import ray

from tools.recipes import audit_run, export_asset


def validate_loss(marker, before, after, evidence, checkpoint, elapsed, bound):
    if not math.isfinite(bound) or bound <= 0 or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Node-loss exit budget and elapsed time must be finite and valid")
    if not marker or marker.get("completed_step") != 4:
        raise ValueError("Node loss was not armed after the complete checkpoint")
    if (
        len(before) != 1
        or any(n.get("Alive") for n in after)
        or not any(n.get("NodeID") == before[0].get("NodeID") for n in after)
    ):
        raise ValueError("The selected worker node was not observed lost")
    if evidence.get("state") != "failed":
        raise ValueError("The failed attempt needs an explicit failed terminal state")
    cleanup = evidence.get("observations", {}).get("cleanup", {})
    if cleanup.get("mode") != "forced_owner_fence" or cleanup.get("state") != "completed":
        raise ValueError("Old attempt owners were not successfully fenced")
    lifecycle = cleanup.get("work_lifecycle", {})
    if lifecycle.get("resident_outputs") != 0 or lifecycle.get("resident_bytes") != 0:
        raise ValueError("Terminal ownership was not proven empty")
    ids = checkpoint.get("committed_work_ids", [])
    if not ids or len(set(ids)) != len(ids) or checkpoint.get("optimizer_step") != 4:
        raise ValueError("Last complete checkpoint is invalid")
    if elapsed > bound:
        raise ValueError("Node-loss failure exceeded its bounded exit budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fault", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-ip", required=True)
    parser.add_argument("--cluster-address", required=True)
    parser.add_argument(
        "--config", default="configs/recipes/qwen3_8b_4k_multinode_node_local_nvlink.yaml"
    )
    parser.add_argument("--exit-bound-seconds", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.exit_bound_seconds) or args.exit_bound_seconds <= 0:
        parser.error("--exit-bound-seconds must be finite and positive")
    from tools.validation.recipe_gpu_smoke import AuditedTrainerActor
    from tools.validation.recipe_gpu_smoke import main as run

    if args.cluster_address in {"local", "auto"}:
        parser.error("Use the explicit address of a dedicated owned cluster")
    root = args.output.resolve()
    if args.phase == "fault":
        root.mkdir(parents=True, exist_ok=False)
    else:
        previous = json.loads((root / "node-loss-fault.json").read_text())
        if previous["status"] != "passed":
            raise ValueError("Recovery requires a passed node-loss/fencing audit")
    os.environ["RAY_ADDRESS"] = args.cluster_address
    os.environ["ASPEC_OUTPUT_DIR"] = str(root)
    os.environ["ASPEC_CACHE_DIR"] = str(root / "cache")
    os.environ.setdefault(
        "ASPEC_DRAFT_CONFIG",
        str(Path(__file__).resolve().parents[2] / "aspec/config/dspark_draft_config.json"),
    )
    ray.init(address=args.cluster_address)
    before = [
        n for n in ray.nodes() if n["NodeManagerAddress"] == args.worker_ip and n.get("Alive")
    ]
    ray.shutdown()
    if len(before) != 1:
        raise ValueError("Exactly one live selected worker is required before the phase")
    if args.phase == "resume" and (
        before[0]["NodeID"] == previous["before_node_id"] or args.worker_ip != previous["worker_ip"]
    ):
        raise ValueError("Recovery requires the selected worker to rejoin as a new node identity")
    output = root / ("first" if args.phase == "fault" else "resumed")
    base = [
        "training.num_train_steps=8",
        "training.lr_total_steps=8",
        "training.num_epochs=100",
        "training.draft_accumulation_steps=1",
        "training.save_interval=4",
        "training.save_final_checkpoint=true",
        "dataset.eval_interval=4",
        "training.optimizer_step_timeout_seconds=120",
        f"cache_dir={root / 'cache'}",
    ]
    if args.phase == "resume":
        base.append(f"training.load_path={root / 'first/checkpoints'}")
    sys.argv = [
        sys.argv[0],
        "--config",
        args.config,
        "--output",
        str(output),
        "--gpus",
        "16",
        "--cluster-address",
        args.cluster_address,
        *base,
    ]

    class ArmedTrainer(AuditedTrainerActor):
        def train_from_queue(self, step, num_batches):
            if self._rank == 0 and step == 4:
                checkpoint = Path(self.args.output_dir) / "checkpoints/iter_0000005/complete.json"
                if not checkpoint.is_file():
                    raise RuntimeError("Refusing to arm without the published checkpoint")
                marker = Path(self.args.output_dir) / "node-loss-armed.json"
                staging = marker.with_suffix(".tmp")
                staging.write_text(
                    json.dumps(
                        {
                            "completed_step": step,
                            "worker_ip": args.worker_ip,
                            "armed_monotonic": time.monotonic(),
                            "armed_unix_seconds": time.time(),
                            "run_id": self.args.aspec_run_id,
                        },
                        indent=2,
                    )
                )
                staging.replace(marker)
                # Hold rank 0 before its next collective/optimizer update while
                # an external controller removes the whole worker node.
                time.sleep(120)
            return super().train_from_queue(step, num_batches)

    result = {
        "phase": args.phase,
        "scope": "externally injected whole worker-node loss and exact recovery",
        "status": "running",
        "worker_ip": args.worker_ip,
        "before_node_id": before[0]["NodeID"],
    }
    try:
        if args.phase == "fault":
            failed = False
            try:
                run(trainer_class=ArmedTrainer)
            except Exception as error:
                failed = True
                result["observed_error"] = f"{type(error).__name__}: {error}"
            if not failed:
                raise ValueError(
                    "Node loss was not injected; healthy completion cannot pass this gate"
                )
            marker_path = output / "node-loss-armed.json"
            marker = json.loads(marker_path.read_text()) if marker_path.is_file() else None
            ray.init(address=args.cluster_address)
            while True:
                after = [n for n in ray.nodes() if n["NodeManagerAddress"] == args.worker_ip]
                observed_dead = any(
                    n["NodeID"] == before[0]["NodeID"] and not n.get("Alive") for n in after
                ) and not any(n.get("Alive") for n in after)
                if (
                    observed_dead
                    or marker is None
                    or time.monotonic() - marker["armed_monotonic"] >= args.exit_bound_seconds
                ):
                    break
                # Actor death can reach the driver before GCS publishes node death.
                # Keep this observation inside the same finite fault-exit budget.
                time.sleep(0.5)
            ray.shutdown()
            evidence = json.loads((output / "evidence.json").read_text())
            work = json.loads((output / "checkpoints/iter_0000005/work_state.json").read_text())
            elapsed = time.monotonic() - marker["armed_monotonic"] if marker else float("inf")
            validate_loss(marker, before, after, evidence, work, elapsed, args.exit_bound_seconds)
            result.update(
                status="passed",
                exit_seconds_after_arm=elapsed,
                dead_node_ids=[n["NodeID"] for n in after if not n.get("Alive")],
                checkpoint_committed_work=len(work["committed_work_ids"]),
            )
        else:
            run()
            audit_run(output, 8, resumed=True, require_eval=True, expected_ranks=8)
            first = json.loads((root / "first/evidence.json").read_text())
            second = json.loads((output / "evidence.json").read_text())
            if first["run_id"] == second["run_id"] or first["attempt_id"] == second["attempt_id"]:
                raise ValueError("Recovery must use a new fenced attempt identity")
            export_asset(output / "checkpoints/iter_0000009", root / "asset")
            result.update(status="passed", recovered_step=8)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (root / f"node-loss-{args.phase}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
