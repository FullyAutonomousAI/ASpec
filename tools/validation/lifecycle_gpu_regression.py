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

"""Supervised real-GPU lifecycle regression; never a performance benchmark."""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

import ray

from aspec.commands._checkpoint_audit import audit_state as audit_state
from aspec.controller.shared_source import SharedSourceState, source_control
from aspec.inference.engine.sgl_engine import SglEngine
from aspec.training.trainer_actor import TrainerActor


class ProbedSglEngine(SglEngine):
    async def shutdown(self):
        # The failed-session path may not collect a healthy feature contract.
        # Observe the real scheduler-owned ring before process teardown instead
        # of treating a coordinator's zero counter as storage-release evidence.
        failure = None
        try:
            if self._engine is not None:
                status = await self.prepare_feature_contract_status()
                path = Path(self.args.output_dir) / f"gpu-source-terminal-{self.rank}.json"
                path.write_text(json.dumps(status, sort_keys=True))
        except Exception as exc:
            failure = exc
        finally:
            await super().shutdown()
        if failure is not None:
            raise failure


class RegressionTrainerActor(TrainerActor):
    def init(self, args, role, mooncake_config=None, with_ref=False):
        result = super().init(args, role, mooncake_config, with_ref)
        self._regression_root = Path(args.output_dir)
        self._regression_scenario = os.environ["ASPEC_LIFECYCLE_SCENARIO"]
        (self._regression_root / "gpu-audit-initial.json").write_text(
            json.dumps(audit_state(self._trainer), sort_keys=True)
        )
        return result

    def save_model(self, step, force_sync=False):
        super().save_model(step, force_sync)
        (self._regression_root / f"gpu-audit-checkpoint-{step:07d}.json").write_text(
            json.dumps(audit_state(self._trainer), sort_keys=True)
        )

    def train_from_queue(self, step, num_batches):
        if (self._regression_scenario == "native_hang"
                and self._regression_root.name == "draft_b" and step == 4):
            import ctypes

            transfer = self._trainer.mooncake_store._p2p_transfer

            def blocked_read(descriptor):
                # An actual admitted GPU descriptor is pending. This is a real
                # unreturning libc call holding the GIL, not Python sleep or a
                # mocked timeout. Only the owning process can end this read.
                (self._regression_root / "native-hang.json").write_text(json.dumps({
                    "pid": os.getpid(), "time": time.time(),
                    "descriptor": descriptor.to_key(), "injection": "libc_read_holds_gil",
                }))
                read_fd, _write_fd = os.pipe()
                library = ctypes.PyDLL(None)
                buffer = ctypes.create_string_buffer(1)
                while True:
                    library.read(read_fd, buffer, 1)

            transfer.read = blocked_read
        if (self._regression_scenario == "cancel_blocked"
                and self._regression_root.name == "draft_b" and step == 4):
            fetcher = self._trainer.data_fetcher
            fetcher = getattr(fetcher, "inner", fetcher)
            observer = fetcher._dataloader.dataset._lifecycle_observer
            ray.get(observer.hub.request_cancel.remote(
                observer.name, "Lifecycle regression: cancel while native step does not return"
            ))
            time.sleep(600)  # Must be fenced by the unchanged driver.
        return super().train_from_queue(step, num_batches)


class RegressionSourceState(SharedSourceState):
    scenario = "train"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.injection = None

    def get_prompts(self, count):
        if self.scenario == "source_fail" and self._published >= 8:
            self.injection = {"kind": "producer_failure", "published_before": self._published}
            self.set_inference_error("Lifecycle regression: injected shared producer failure")
            raise RuntimeError("Lifecycle regression: injected shared producer failure")
        return super().get_prompts(count)

    def _confirm_subscription(self, execution_id):
        if (self.scenario in {"subscribe_rejected", "subscribe_ack_lost"}
                and self._published >= 48 and self.injection is None):
            descriptor, audience = self._pending_subscriptions[execution_id]
            # Same real GPU source/control endpoint. For lost-ACK simulation,
            # perform idempotent registrations but discard their returned result
            # at the caller, exactly at the ambiguous RPC completion boundary.
            if self.scenario == "subscribe_ack_lost":
                for _ in range(3):
                    source_control(descriptor, operation="subscribe", readers=list(audience))
            self.injection = {
                "kind": self.scenario,
                "published_before": self._published,
                "readers": list(audience),
                "slot_id": descriptor.slot_id,
                "generation": descriptor.generation,
                "registration_completed_on_real_source": self.scenario == "subscribe_ack_lost",
                "outstanding_at_fault": self.get_pool_size(),
                "pending_subscriptions_at_fault": len(self._pending_subscriptions),
            }
            raise RuntimeError(f"Lifecycle regression: injected {self.scenario}")
        return super()._confirm_subscription(execution_id)

    def status(self):
        result = super().status()
        result["p1_fault_injection"] = self.injection
        return result


class RejectedSubscriptionSourceState(RegressionSourceState):
    scenario = "subscribe_rejected"


class LostAckSubscriptionSourceState(RegressionSourceState):
    scenario = "subscribe_ack_lost"


class ProducerFailureSourceState(RegressionSourceState):
    scenario = "source_fail"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(
        "train", "cancel_blocked", "native_hang", "source_fail", "subscribe_rejected", "subscribe_ack_lost", "resume"
    ), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    if args.scenario == "resume" and not args.resume_from:
        parser.error("--resume-from is required")
    root.mkdir(parents=True)

    # Private per-run credentials and cluster; never join/stop another Ray job.
    token_path = root / "ray-auth-token"
    descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_hex(32))
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token_path)
    os.environ.pop("RAY_AUTH_TOKEN", None)
    os.environ.pop("RAY_ADDRESS", None)
    from aspec import multi_draft, train_entry

    original_compile = multi_draft.compile_multi_draft_plan

    def compile_test(config):
        for experiment in config.experiments:
            experiment.training.train_env_vars = json.dumps({"ASPEC_LIFECYCLE_SCENARIO": args.scenario})
            if args.scenario == "native_hang" and experiment.name == "draft_b":
                # Only accelerate detection of the deliberately hung reader.
                # The healthy reader retains its normal wall-time budget: a
                # shared bounded pool can backpressure it until B is fenced.
                experiment.training.optimizer_step_timeout_seconds = 90
        return original_compile(config)

    multi_draft.compile_multi_draft_plan = compile_test
    multi_draft.TrainerActor = RegressionTrainerActor
    from aspec.inference.engine import sgl_engine

    sgl_engine.SglEngine = ProbedSglEngine
    source_class = {
        "source_fail": ProducerFailureSourceState,
        "subscribe_rejected": RejectedSubscriptionSourceState,
        "subscribe_ack_lost": LostAckSubscriptionSourceState,
    }.get(args.scenario, RegressionSourceState)
    multi_draft.SharedSourceController = ray.remote(max_concurrency=32)(source_class)
    experiments = [
        {"name": "draft_a", "training": {"seed": 42, "num_train_steps": 16}},
        {"name": "draft_b", "training": {
            "seed": 43, "num_train_steps": 20, "learning_rate": 0.0003, "min_lr": 0.00003
        }},
    ]
    if args.scenario == "resume":
        for experiment in experiments:
            old = Path(args.resume_from) / experiment["name"] / "checkpoints"
            iteration = int((old / "latest_checkpointed_iteration.txt").read_text())
            checkpoint = old / f"iter_{iteration:07d}"
            step = json.loads((checkpoint / "work_state.json").read_text())["optimizer_step"]
            experiment["training"]["load_path"] = str(checkpoint)
            # Resume the canceled reader for sixteen updates. The already
            # completed reader explicitly exercises the no-new-work branch.
            experiment["training"]["num_train_steps"] = step if experiment["name"] == "draft_a" else step + 16
    sys.argv = [
        sys.argv[0], "--config", "configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml",
        "inference.inference_num_gpus=1", "inference.inference_num_gpus_per_node=3",
        "training.training_num_gpus_per_node=1", "training.lr_total_steps=40",
        "training.save_interval=" + ("4" if args.scenario in {"cancel_blocked", "native_hang"} else "0"),
        "training.max_checkpoints=2", "training.save_final_checkpoint=true",
        "training.continual_training=false",
        "debug.performance_measurement_required=false",
        "experiments=" + json.dumps(experiments),
        "output_dir=" + str(root),
        "cache_dir=" + str(root.parent / "cache"),
    ]
    (root / "gpu-regression-command.json").write_text(json.dumps({
        "diagnostic_only": True, "scenario": args.scenario, "argv": sys.argv,
        "python": sys.executable, "driver_pid": os.getpid(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "resume_from": args.resume_from, "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }, indent=2))
    ray.init(
        address="local", num_cpus=16, num_gpus=3, include_dashboard=False,
        object_store_memory=1024 ** 3, _temp_dir=f"/tmp/aspec-p1-ray-{uuid.uuid4().hex[:10]}"
    )
    try:
        train_entry.main()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
