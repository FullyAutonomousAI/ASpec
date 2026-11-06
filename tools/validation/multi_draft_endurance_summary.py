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

"""Check native long-run lifecycle evidence and summarize memory observations."""

import argparse
import json
from pathlib import Path

from aspec.reporting import build_run_report


def summarize(root, *, healthy="draft_a", failed="draft_b", slow_steps, fail_step):
    root = Path(root)
    evidence = json.loads((root / "evidence.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "endurance-observations.jsonl").read_text().splitlines()
    ]
    observations = [row for row in rows if "source" in row]
    if not observations:
        raise ValueError("no usable long-run observations")
    source = evidence["shared_source"]
    if evidence["state"] != "completed_with_failed_experiments":
        raise ValueError("long run did not isolate the injected Draft failure")
    if evidence["cleanup_errors"] or evidence["read_owners_fenced"] is not True:
        raise ValueError("long-run cleanup did not safely fence read owners")
    if (
        source["outstanding_observations"]
        or source["outstanding_payload_bytes"]
        or source["pending_production"]
    ):
        raise ValueError("shared source did not reach a storage-free terminal state")
    if source["retiring_experiments"] or source["active_experiments"]:
        raise ValueError("shared experiment retirement is incomplete")
    if evidence["inference_manager"]["inflight_samples"]:
        raise ValueError("producer inflight work remains")
    health = build_run_report(root / healthy)
    if health["health"] != "healthy":
        raise ValueError("surviving experiment is not healthy")
    if evidence["experiments"][failed]["state"] != "failed":
        raise ValueError("injected experiment failure is missing")
    if (
        "injected isolated failure after slow-consumer recovery"
        not in evidence["experiments"][failed]["error"]
    ):
        raise ValueError("experiment failed for a reason other than the intended injection")
    checkpoints = {}
    for name in (healthy, failed):
        base = root / name / "checkpoints"
        iteration = int((base / "latest_checkpointed_iteration.txt").read_text())
        state = json.loads((base / f"iter_{iteration:07d}" / "work_state.json").read_text())
        ids = state["committed_work_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate committed semantic work")
        checkpoints[name] = state
    if checkpoints[failed]["optimizer_step"] != fail_step:
        raise ValueError("failure checkpoint is not at the expected committed frontier")
    if not set(checkpoints[failed]["committed_work_ids"]).issubset(
        checkpoints[healthy]["committed_work_ids"]
    ):
        raise ValueError("shared logical work differs between experiments")
    progress = {name: [] for name in (healthy, failed)}
    for row in observations:
        for name in progress:
            child = row["source"]["experiments"][name]
            progress[name].append(child["lifecycle"]["optimizer_steps_committed"])
    if any(values != sorted(values) for values in progress.values()):
        raise ValueError("committed progress regressed during the long run")
    recovery = [
        row
        for row in observations
        if slow_steps
        <= row["source"]["experiments"][failed]["lifecycle"]["optimizer_steps_committed"]
        < fail_step
    ]
    if not recovery:
        raise ValueError("consumer recovery phase was not observed")
    rss = {}
    for row in observations:
        for process in row["processes"]:
            if (
                "ObservedSourceState" in process["name"]
                or "EnduranceTrainerActor" in process["name"]
            ):
                values = rss.setdefault(
                    str(process["pid"]), {"name": process["name"], "samples": []}
                )
                values["samples"].append((row["elapsed_seconds"], process["rss_bytes"]))
    for value in rss.values():
        samples = value.pop("samples")
        value.update(
            {
                "first_rss_bytes": samples[0][1],
                "last_rss_bytes": samples[-1][1],
                "peak_rss_bytes": max(row[1] for row in samples),
                "samples": len(samples),
            }
        )
    return {
        "diagnostic_only": True,
        "session_wall_seconds": evidence["wall_seconds"],
        "observed_seconds": observations[-1]["elapsed_seconds"]
        - observations[0]["elapsed_seconds"],
        "usable_observations": len(observations),
        "observation_errors": len(rows) - len(observations),
        "shared_source": source,
        "inference_manager": evidence["inference_manager"],
        "checkpoint_progress": {
            name: {
                "optimizer_step": state["optimizer_step"],
                "committed_work": len(state["committed_work_ids"]),
                "dataset_epoch": state["dataset_epoch"],
            }
            for name, state in checkpoints.items()
        },
        "peak_outstanding_observations": max(
            row["source"]["outstanding_observations"] for row in observations
        ),
        "peak_tracked_epochs": {
            name: max(row["source"]["experiments"][name]["tracked_epochs"] for row in observations)
            for name in progress
        },
        "maximum_step_lead_during_slow_phase": max(
            row["source"]["experiments"][healthy]["lifecycle"]["optimizer_steps_committed"]
            - row["source"]["experiments"][failed]["lifecycle"]["optimizer_steps_committed"]
            for row in observations
            if row["source"]["experiments"][failed]["lifecycle"]["optimizer_steps_committed"]
            < slow_steps
        ),
        "process_rss": rss,
        "surviving_experiment_health": health["health"],
        "all_committed_work_unique": True,
        "shared_committed_prefix_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--slow-steps", type=int, required=True)
    parser.add_argument("--fail-step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.session, slow_steps=args.slow_steps, fail_step=args.fail_step)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
