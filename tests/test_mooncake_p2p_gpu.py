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

# ruff: noqa: I001

"""Opt-in two-GPU Mooncake NVLink transfer and lease-lifecycle check."""

import os

import pytest

if os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1":
    pytest.skip(
        "set ASPEC_RUN_GPU_INTEGRATION=1 on a two-GPU NVLink host",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist

from aspec.config.mooncake_config import MooncakeConfig
from aspec.transfer.mooncake.p2p import (
    MooncakeP2PDescriptor,
    MooncakeP2PTransfer,
)

def _config():
    return MooncakeConfig(
        local_hostname="127.0.0.1",
        data_plane="transfer_engine",
        protocol="nvlink_intra",
        require_nvlink=True,
        enable_gpu_direct=True,
        allow_host_fallback=False,
        gpu_buffer_size=2 * 1024 * 1024,
        gpu_ring_slots=1,
        get_retry_max_wait_seconds=10,
    )


def test_nvlink_transfer_moves_cuda_tensors_and_releases_both_rings():
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch with torchrun --standalone --nproc-per-node=2")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    transfer = MooncakeP2PTransfer(_config())
    try:
        role = "source" if local_rank == 0 else "destination"
        transfer.setup(torch.device("cuda", local_rank), role=role)
        generations = []
        for round_id in range(2):
            descriptor_key = [None]
            if local_rank == 0:
                hidden = (
                    torch.arange(
                        32 * 64,
                        device=f"cuda:{local_rank}",
                        dtype=torch.bfloat16,
                    ).reshape(32, 64)
                    + round_id
                )
                input_ids = (
                    torch.arange(
                        32, device=f"cuda:{local_rank}", dtype=torch.int64
                    )
                    + round_id
                )
                descriptor = transfer.stage(
                    store_key=f"gpu-integration-{round_id}",
                    tensors=(("hidden_states", hidden), ("input_ids", input_ids)),
                    run_id="gpu-integration",
                    attempt_id="gpu-integration-attempt",
                    data_id=f"sample-{round_id}",
                    request_id=f"request-{round_id}",
                    source_engine="0",
                )
                descriptor_key[0] = descriptor.to_key()

            dist.broadcast_object_list(descriptor_key, src=0)
            descriptor = MooncakeP2PDescriptor.from_key(descriptor_key[0])
            generations.append(descriptor.generation)
            if local_rank == 0:
                assert transfer.wait_for_source_releases(timeout=10)
                assert transfer.ring.in_use == 0
            else:
                received = transfer.read(descriptor)
                expected_hidden = (
                    torch.arange(
                        32 * 64,
                        device=f"cuda:{local_rank}",
                        dtype=torch.bfloat16,
                    ).reshape(32, 64)
                    + round_id
                )
                expected_ids = (
                    torch.arange(
                        32, device=f"cuda:{local_rank}", dtype=torch.int64
                    )
                    + round_id
                )
                torch.testing.assert_close(
                    received.tensors["hidden_states"], expected_hidden
                )
                torch.testing.assert_close(
                    received.tensors["input_ids"], expected_ids
                )
                received.lease.release()
                with pytest.raises(RuntimeError, match="stale or duplicate"):
                    received.lease.release()
                assert transfer.ring.in_use == 0
            dist.barrier()

        assert generations[1] == generations[0] + 1
    finally:
        transfer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("operation", ["stage", "discard"])
def test_source_rollback_waits_for_queued_cuda_writes(operation):
    from aspec.transfer.mooncake.p2p import GenerationSlotRing

    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    # Exercise actual CUDA stream ordering independently of transport setup.
    transfer = MooncakeP2PTransfer(_config())
    transfer.role, transfer.device = "source", device
    transfer.endpoint = "rollback-test:1"
    transfer.ring = GenerationSlotRing(1)
    transfer.buffers = [torch.empty(1024, dtype=torch.uint8, device=device)]
    transfer.copy_stream = torch.cuda.Stream(device=device)
    written = torch.cuda.Event()
    value = torch.ones(16, device=device)
    try:
        if operation == "stage":
            with torch.cuda.stream(transfer.copy_stream):
                torch.cuda._sleep(200_000_000)

            def interrupted_tensors():
                yield "hidden_states", value
                written.record(transfer.copy_stream)
                raise RuntimeError("interrupted stage")

            with pytest.raises(RuntimeError, match="interrupted stage"):
                transfer.stage(
                    store_key="rollback", tensors=interrupted_tensors(),
                    run_id="run", attempt_id="attempt",
                )
        else:
            reservation = transfer.reserve(
                store_key="rollback", tensors=[("hidden_states", (16,), torch.float32)],
                run_id="run", attempt_id="attempt",
            )
            producer = torch.cuda.Stream(device=device)
            producer.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(producer):
                torch.cuda._sleep(200_000_000)
                reservation.tensors["hidden_states"].copy_(value)
                written.record(producer)
            reservation.discard()
        assert written.query(), "slot returned before its CUDA writes completed"
        assert transfer.ring.in_use == 0
    finally:
        # Failure on a regressed implementation must not free live GPU storage.
        torch.cuda.synchronize(device)
        transfer.close()
