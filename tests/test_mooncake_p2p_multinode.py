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

"""Explicit two-node generation-checked TransferEngine integration test.

Run against an already-started two-node Ray cluster. Replace the RFC 5737
documentation addresses below with the two node addresses:

    ASPEC_RUN_MULTI_NODE_GPU_INTEGRATION=1 \
    ASPEC_MULTI_NODE_SOURCE_IP=192.0.2.32 \
    ASPEC_MULTI_NODE_DESTINATION_IP=198.51.100.20 \
    ASPEC_MULTI_NODE_PROTOCOL=tcp \
    RAY_ADDRESS=198.51.100.20:6379 \
    python -m pytest -q tests/test_mooncake_p2p_multinode.py -s
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_MULTI_NODE_GPU_INTEGRATION") != "1",
    reason="set ASPEC_RUN_MULTI_NODE_GPU_INTEGRATION=1 for two-node GPU integration",
)


def test_cross_node_transfer_engine_round_trip():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    source_ip = os.environ["ASPEC_MULTI_NODE_SOURCE_IP"]
    destination_ip = os.environ["ASPEC_MULTI_NODE_DESTINATION_IP"]
    protocol = os.getenv("ASPEC_MULTI_NODE_PROTOCOL", "tcp")
    device_name = os.getenv("ASPEC_MULTI_NODE_RDMA_DEVICE", "")
    rdma_port = int(os.getenv("ASPEC_MULTI_NODE_RDMA_PORT", "1"))
    rdma_gid_index = int(os.getenv("ASPEC_MULTI_NODE_RDMA_GID_INDEX", "3"))
    if protocol not in {"tcp", "rdma"}:
        raise ValueError(f"unsupported integration protocol {protocol!r}")
    if protocol == "rdma" and not device_name:
        raise ValueError("RDMA integration requires ASPEC_MULTI_NODE_RDMA_DEVICE")
    run_id = f"multinode-integration-{uuid.uuid4()}"

    ray.init(address=os.environ["RAY_ADDRESS"])
    alive_nodes = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive")
    }
    assert set(alive_nodes) == {source_ip, destination_ip}

    @ray.remote(num_gpus=1)
    class TransferActor:
        def __init__(
            self,
            role: str,
            transport_protocol: str,
            device_name: str,
            rdma_port: int,
            rdma_gid_index: int,
            expected_run_id: str,
            expected_attempt_id: str,
        ):
            import torch

            from aspec.config.mooncake_config import MooncakeConfig
            from aspec.transfer.mooncake.p2p import MooncakeP2PTransfer

            os.environ["ASPEC_RUN_ID"] = expected_run_id
            os.environ["ASPEC_ATTEMPT_ID"] = expected_attempt_id
            self.role = role
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
            config = MooncakeConfig(
                local_hostname=ray.util.get_node_ip_address(),
                protocol=transport_protocol,
                device_name=device_name,
                rdma_port=rdma_port,
                rdma_gid_index=rdma_gid_index,
                data_plane="transfer_engine",
                gpu_buffer_size=1024 * 1024,
                gpu_ring_slots=1,
                enable_gpu_direct=True,
                allow_host_fallback=False,
                require_nvlink=False,
                get_retry_max_wait_seconds=30,
            )
            config.export_env()
            self.transfer = MooncakeP2PTransfer(config)
            self.transfer.setup(self.device, role=role)

        def stage(self, expected_run_id: str, expected_attempt_id: str) -> str:
            import torch

            assert self.role == "source"
            hidden = torch.arange(4096, dtype=torch.float32, device=self.device).reshape(
                128, 32
            )
            tokens = torch.arange(128, dtype=torch.int64, device=self.device)
            descriptor = self.transfer.stage(
                store_key="multinode-smoke",
                tensors=(
                    ("hidden_states", hidden),
                    ("input_ids", tokens),
                ),
                run_id=expected_run_id,
                attempt_id=expected_attempt_id,
                data_id="source-0",
                request_id="request-0",
                source_engine="source-node",
            )
            return descriptor.to_key()

        def read_and_validate(self, descriptor_key: str) -> dict:
            import torch

            from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor

            assert self.role == "destination"
            descriptor = MooncakeP2PDescriptor.from_key(descriptor_key)
            result = self.transfer.read(descriptor)
            expected_hidden = torch.arange(
                4096, dtype=torch.float32, device=self.device
            ).reshape(128, 32)
            expected_tokens = torch.arange(128, dtype=torch.int64, device=self.device)
            torch.cuda.synchronize(self.device)
            matched = bool(
                torch.equal(result.tensors["hidden_states"], expected_hidden)
                and torch.equal(result.tensors["input_ids"], expected_tokens)
            )
            result.lease.release()
            return {
                "matched": matched,
                "payload_bytes": descriptor.payload_bytes,
                "source_endpoint": descriptor.source_endpoint,
                "run_id": descriptor.run_id,
                "destination_ip": ray.util.get_node_ip_address(),
            }

        def read_expect_failure(self, descriptor_key: str) -> dict:
            from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor

            assert self.role == "destination"
            descriptor = MooncakeP2PDescriptor.from_key(descriptor_key)
            started = time.monotonic()
            try:
                result = self.transfer.read(descriptor)
            except BaseException as exc:
                return {
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": self.transfer.status(),
                }
            result.lease.release()
            return {
                "elapsed_seconds": time.monotonic() - started,
                "error": None,
                "status": self.transfer.status(),
            }

        def wait_for_release(self) -> bool:
            return self.transfer.wait_for_source_releases(timeout=30)

        def status(self) -> dict:
            return self.transfer.status()

        def crash_owner_process(self) -> None:
            # ``ray.kill(actor)`` invalidates the actor handle but may keep the
            # GPU worker process (and its native TransferEngine endpoint)
            # alive for reuse.  Exit inside the actor so this test actually
            # fences the process that owns the registered source memory.
            os._exit(86)

        def close(self) -> None:
            self.transfer.close()

    attempt_id = f"multinode-attempt-{uuid.uuid4()}"
    source = TransferActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=alive_nodes[source_ip], soft=False
        )
    ).remote(
        "source",
        protocol,
        device_name,
        rdma_port,
        rdma_gid_index,
        run_id,
        attempt_id,
    )
    destination = TransferActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=alive_nodes[destination_ip], soft=False
        )
    ).remote(
        "destination",
        protocol,
        device_name,
        rdma_port,
        rdma_gid_index,
        run_id,
        attempt_id,
    )

    try:
        descriptor_key = ray.get(
            source.stage.remote(run_id, attempt_id), timeout=60
        )
        result = ray.get(
            destination.read_and_validate.remote(descriptor_key), timeout=60
        )
        assert result["matched"] is True
        assert result["payload_bytes"] > 0
        assert result["run_id"] == run_id
        assert result["destination_ip"] == destination_ip
        assert ray.get(source.wait_for_release.remote(), timeout=60) is True
        source_status, destination_status = ray.get(
            [source.status.remote(), destination.status.remote()], timeout=60
        )
        assert source_status["protocol"] == protocol
        assert destination_status["protocol"] == protocol
        assert destination_status["strict_transport"] is True
        assert destination_status["metrics"]["destination_reads"] == 1
        assert destination_status["metrics"]["destination_payload_bytes"] > 0
        assert destination_status["metrics"]["transfer_failures"] == 0

        # Stage a new generation, then abruptly kill its owner process before
        # the destination reads.  Both TCP and RDMA must fail explicitly and
        # release destination credit rather than accepting stale data.  A
        # graceful transfer.close() is deliberately insufficient here: it does
        # not exercise process fencing or loss of registered-memory ownership.
        failed_descriptor_key = ray.get(
            source.stage.remote(run_id, attempt_id), timeout=60
        )
        source.crash_owner_process.remote()
        death_deadline = time.monotonic() + 30
        while True:
            try:
                ray.get(source.status.remote(), timeout=1)
            except ray.exceptions.GetTimeoutError:
                if time.monotonic() >= death_deadline:
                    pytest.fail("source owner process did not terminate within 30 seconds")
                continue
            except ray.exceptions.RayActorError:
                break
            if time.monotonic() >= death_deadline:
                pytest.fail("source owner remained callable after ray.kill")
            time.sleep(0.05)
        source = None
        failure = ray.get(
            destination.read_expect_failure.remote(failed_descriptor_key), timeout=60
        )
        assert failure["error"] is not None
        assert failure["elapsed_seconds"] < 60
        assert failure["status"]["metrics"]["transfer_failures"] == 1
        assert failure["status"]["ring"]["in_use"] == 0
    finally:
        close_refs = [destination.close.remote()]
        if source is not None:
            close_refs.append(source.close.remote())
        ray.get(close_refs, timeout=60)
        if source is not None:
            ray.kill(source, no_restart=True)
        ray.kill(destination, no_restart=True)
        ray.shutdown()
