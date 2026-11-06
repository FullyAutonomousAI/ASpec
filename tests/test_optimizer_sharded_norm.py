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

"""Opt-in distributed GPU regression; ordinary pytest does not spawn processes.

ASPEC_RUN_GPU_INTEGRATION=1 torchrun --standalone --nproc-per-node=2 \
    -m pytest -q tests/test_optimizer_sharded_norm.py
"""

import os

import pytest

if os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or "LOCAL_RANK" not in os.environ:
    pytest.skip("requires an explicitly launched distributed GPU test", allow_module_level=True)


def test_optimizer_returns_global_not_local_shard_gradient_norm():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    from aspec.training.optimizer import BF16Optimizer

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        model = torch.nn.Module()
        full = torch.arange(1, 1025, device="cuda", dtype=torch.float32).div(1024)
        model.weight = torch.nn.Parameter(
            distribute_tensor(
                torch.ones(1024, device="cuda", dtype=torch.bfloat16), mesh, (Shard(0),)
            )
        )
        optimizer = BF16Optimizer(
            model, lr=0.001, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0
        )
        gradient = full.to(torch.bfloat16)
        expected = torch.linalg.vector_norm(gradient.float())
        model.weight.grad = distribute_tensor(gradient, mesh, (Shard(0),))
        reported = optimizer.step()
        assert not isinstance(reported, DTensor)
        torch.testing.assert_close(reported, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            optimizer.model_params[0].full_tensor(),
            optimizer.fp32_params[0].full_tensor().to(torch.bfloat16),
        )
    finally:
        dist.destroy_process_group()
