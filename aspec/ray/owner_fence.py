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

"""Observe read-process exit before retiring attempt-local output ownership.

The process probe is retained from ASpec's shared-Draft fencing path.
Lost-node acceptance is limited to declared node-local fail-stop domains;
it does not certify physical memory reclamation on an unreachable host.
"""

import time
from collections import defaultdict

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aspec.runtime_control import termination_signal_handlers
from aspec.utils.logging import logger


class SourceTeardownBlockedError(RuntimeError):
    """Keep unverified reader ownership visible across entry-point cleanup."""

    def __init__(self, owner_statuses, *, allow_node_loss=False, cleanup=None):
        super().__init__("Source teardown blocked: read-owner process exit is unverified")
        self.owner_statuses = owner_statuses
        self.allow_node_loss = allow_node_loss
        self.cleanup = cleanup


def run_with_source_fencing(entrypoint):
    """Keep a CLI driver alive until Ray's implicit exit cleanup is safe.

    Each probe is bounded. A failed fence requires external intervention;
    returning from the CLI would let Ray's atexit hook reclaim source actors.
    Driver/host loss and forced process termination remain outside this guard.
    """
    try:
        return entrypoint()
    except SourceTeardownBlockedError as blocked:
        retry = False
        with termination_signal_handlers(defer=True):
            logger.error(
                "Source teardown blocked; retaining the Ray driver until read owners "
                "are fenced. Do not terminate the driver or cluster before external fencing."
            )
            while True:
                try:
                    if retry:
                        time.sleep(1)
                    wait_for_owner_exit(
                        blocked.owner_statuses, allow_node_loss=blocked.allow_node_loss,
                    )
                    break
                except (Exception, KeyboardInterrupt) as exc:
                    retry = True
                    logger.error("Read-owner exit remains unverified; driver retained: %s", exc)
        if ray.is_initialized():
            ray.shutdown()
        # Preserve the failed run and its original error chain after fencing.
        raise


@ray.remote(num_cpus=0)
class _OwnerFenceProbe:
    def exited(self, pids):
        import psutil

        for pid in pids:
            try:
                if psutil.Process(pid).status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return False
            except psutil.NoSuchProcess:
                pass
        return True


def wait_for_owner_exit(statuses, *, allow_node_loss=False, timeout=30):
    if not isinstance(statuses, list) or not statuses:
        raise ValueError("Missing initialized read-owner process identities")
    for status in statuses:
        if (
            not isinstance(status, dict)
            or not status.get("node_id")
            or not isinstance(status.get("pid"), int)
            or status["pid"] <= 0
        ):
            raise ValueError("Invalid initialized read-owner process identity")
    groups = defaultdict(list)
    for status in statuses:
        groups[status["node_id"]].append(status)
    deadline = time.monotonic() + timeout
    lost = []
    exited = 0
    for node_id, owners in groups.items():
        probe = None
        pending = None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Read-owner processes did not exit before fencing deadline")
                node = next((n for n in ray.nodes() if n["NodeID"] == node_id), None)
                if node is None or any(
                    node.get("NodeManagerAddress") != o.get("node_ip") for o in owners
                ):
                    raise RuntimeError("Read-owner node identity is unknown or changed")
                if not node.get("Alive"):
                    if not allow_node_loss:
                        raise RuntimeError("Cannot prove remote read-process exit on a lost node")
                    lost.append(node_id)
                    break
                if probe is None:
                    probe = _OwnerFenceProbe.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=node_id, soft=False
                        )
                    ).remote()
                if pending is None:
                    pending = probe.exited.remote([owner["pid"] for owner in owners])
                try:
                    if ray.get(pending, timeout=min(1, remaining)):
                        exited += len(owners)
                        break
                    pending = None
                except ray.exceptions.GetTimeoutError:
                    pass
                except (ray.exceptions.ActorDiedError, ray.exceptions.ActorUnavailableError):
                    pending = None
                time.sleep(0.05)
        finally:
            if probe is not None:
                ray.kill(probe, no_restart=True)
    return {
        "state": "verified",
        "scope": "process exit on live nodes; node-local fail-stop domain loss",
        "live_node_processes": exited,
        "lost_node_ids": sorted(set(lost)),
        "physical_memory_unobserved_nodes": sorted(set(lost)),
    }
