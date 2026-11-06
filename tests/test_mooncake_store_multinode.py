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

"""Opt-in two-node Store/TCP byte correctness and buffer reuse probe.

Uses the same RAY_ADDRESS, ASPEC_MULTI_NODE_SOURCE_IP,
ASPEC_MULTI_NODE_DESTINATION_IP and ASPEC_RUN_MULTI_NODE_GPU_INTEGRATION
environment variables as test_mooncake_p2p_multinode.py. Not a default unit test.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_MULTI_NODE_GPU_INTEGRATION") != "1",
    reason="explicit two-node GPU integration only",
)


def test_host_staged_tcp_round_trip_and_buffer_reuse():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from aspec.ray.ray_actor import RayActor
    from aspec.transfer.mooncake.utils import MooncakeMaster

    ray.init(address=os.environ["RAY_ADDRESS"])
    nodes = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive")
    }
    source_ip = os.environ["ASPEC_MULTI_NODE_SOURCE_IP"]
    destination_ip = os.environ["ASPEC_MULTI_NODE_DESTINATION_IP"]
    assert source_ip != destination_ip
    assert {source_ip, destination_ip} <= set(nodes)

    @ray.remote(num_gpus=1)
    class StoreEndpoint:
        def __init__(self, role, address, metadata_port):
            import torch

            from aspec.config.mooncake_config import MooncakeConfig
            from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
            self.store = EagleMooncakeStore(MooncakeConfig(
                local_hostname=ray.util.get_node_ip_address(),
                master_server_address=address,
                metadata_server=f"http://{address.split(':')[0]}:{metadata_port}/metadata",
                data_plane="store",
                protocol="tcp",
                enable_gpu_direct=False,
                allow_host_fallback=False,
                # Mooncake's native segment allocator requires at least 16 MiB.
                global_segment_size="32MB" if role == "source" else 0,
                local_buffer_size="4MB",
                host_buffer_size="2MB",
                async_put_pool_size=1 if role == "source" else 0,
                get_retry_max_wait_seconds=15,
            ))
            self.store.setup(self.device)

        def expected(self, generation):
            import torch

            ids = torch.arange(128, device=self.device, dtype=torch.int64) + generation
            hidden = ids[:, None].expand(128, 32).to(torch.bfloat16)
            return ids, hidden, hidden[:, :8].contiguous()

        def publish(self, key, generation):
            ids, hidden, last = self.expected(generation)
            # Do not flush here: the consumer must handle asynchronous publication.
            if generation == 2:
                return self.store.put_direct_cp_shards(
                    key, hidden, ids, last, cp_size=1, max_seq_length=128
                )
            return self.store.put(key, hidden, ids, last)

        def consume(self, key, metadata, generation):
            import torch

            output = self.store.get(
                key, metadata["shapes"], metadata["dtypes"], self.device
            )
            ids, hidden, last = self.expected(generation)
            assert torch.equal(output.input_ids, ids)
            assert torch.equal(output.hidden_states, hidden)
            assert torch.equal(output.last_hidden_states, last)
            self.store.remove_eagle3_tensors(key, has_last_hidden_states=True)
            remaining = self.store.batch_exists(
                [f"{key}_hs", f"{key}_ids", f"{key}_lhs"]
            )
            assert not any(remaining.values())
            return {"matched": True, "deleted": True, "generation": generation}

        def close(self):
            self.store.close()

    master = ray.remote(num_cpus=0)(MooncakeMaster).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        )
    ).remote()
    endpoints = []
    try:
        info = ray.get(master.start.remote(
            RayActor.find_free_port(start_port=51000),
            RayActor.find_free_port(start_port=8100),
        ), timeout=60)
        for role, ip in (("source", source_ip), ("destination", destination_ip)):
            endpoints.append(StoreEndpoint.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=nodes[ip], soft=False
                )
            ).remote(role, info["master_addr"], info["metadata_port"]))
        source, destination = endpoints
        prefix = f"store-integration-{uuid.uuid4()}"
        for generation in range(3):
            key = f"{prefix}-{generation}"
            metadata = ray.get(source.publish.remote(key, generation), timeout=60)
            result = ray.get(
                destination.consume.remote(key, metadata, generation), timeout=60
            )
            assert result == {"matched": True, "deleted": True, "generation": generation}
    finally:
        try:
            ray.get([endpoint.close.remote() for endpoint in endpoints], timeout=30)
        finally:
            for endpoint in endpoints:
                ray.kill(endpoint, no_restart=True)
            try:
                ray.get(master.shutdown.remote(), timeout=30)
            finally:
                ray.kill(master, no_restart=True)
                ray.shutdown()
