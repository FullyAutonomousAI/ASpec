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

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from aspec.config.mooncake_config import MooncakeConfig, resolve_rdma_routes


def test_flat_args_preserve_explicit_transfer_timeouts():
    args = SimpleNamespace(
        mooncake_local_hostname="worker",
        mooncake_store_full_error_codes=[-200, -201],
        mooncake_store_full_wait_seconds=0.25,
        mooncake_store_full_log_interval_seconds=2.0,
        mooncake_store_full_max_wait_seconds=30.0,
        mooncake_get_retry_wait_seconds=0.75,
        mooncake_get_retry_log_interval_seconds=4.0,
        mooncake_get_retry_max_wait_seconds=300.0,
        mooncake_transfer_timeout_seconds=420,
        mooncake_tcp_progress_timeout_seconds=180,
    )

    with mock.patch("aspec.ray.ray_actor.RayActor.get_node_ip"):
        config = MooncakeConfig.from_flat_args(args)

    assert config.store_full_error_codes == (-200, -201)
    assert config.store_full_wait_seconds == 0.25
    assert config.store_full_log_interval_seconds == 2.0
    assert config.store_full_max_wait_seconds == 30.0
    assert config.get_retry_wait_seconds == 0.75
    assert config.get_retry_log_interval_seconds == 4.0
    assert config.get_retry_max_wait_seconds == 300.0
    assert config.transfer_timeout_seconds == 420
    assert config.tcp_progress_timeout_seconds == 180


def test_transfer_timeouts_export_to_native_mooncake_and_round_trip():
    config = MooncakeConfig(
        protocol="tcp",
        transfer_timeout_seconds=420,
        tcp_progress_timeout_seconds=180,
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        config.export_env()
        assert os.environ["MC_TRANSFER_TIMEOUT"] == "420"
        assert os.environ["MC_TCP_PROGRESS_TIMEOUT_SEC"] == "180"
        restored = MooncakeConfig.from_env()

    assert restored.transfer_timeout_seconds == 420
    assert restored.tcp_progress_timeout_seconds == 180


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transfer_timeout_seconds", 0),
        ("tcp_progress_timeout_seconds", -1),
        ("transfer_timeout_seconds", 1.5),
    ],
)
def test_transfer_timeouts_require_positive_integer_seconds(field, value):
    with pytest.raises(ValueError, match=field):
        MooncakeConfig(**{field: value})


def test_transfer_engine_is_strict_and_uses_exact_per_sample_slot_size():
    config = MooncakeConfig(
        data_plane="transfer_engine",
        protocol="nvlink_intra",
        require_nvlink=True,
        enable_gpu_direct=True,
        allow_host_fallback=False,
        max_seq_len=131072,
        hidden_dim=4096,
        num_aux_layers=5,
    )

    assert config.gpu_buffer_size == 6_443_499_520
    assert config.ring_slots_for_role("source") == 2
    assert config.ring_slots_for_role("destination") == 2


def test_tcp_normalizes_null_device_name_before_export():
    config = MooncakeConfig(protocol="tcp", device_name=None)

    with mock.patch.dict(os.environ, {}, clear=True):
        config.export_env()
        assert config.device_name == ""
        assert os.environ["MOONCAKE_DEVICE_NAME"] == ""


def test_rdma_route_is_exported_and_round_trips_through_environment():
    config = MooncakeConfig(
        protocol="rdma",
        device_name="mlx5_bond_0",
        rdma_port=1,
        rdma_gid_index=3,
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        config.export_env()
        assert os.environ["MC_IB_PORT"] == "1"
        assert os.environ["MC_GID_INDEX"] == "3"
        assert os.environ["MOONCAKE_RDMA_GID_INDEX"] == "3"
        restored = MooncakeConfig.from_env()

    assert restored.device_name == "mlx5_bond_0"
    assert restored.rdma_port == 1
    assert restored.rdma_gid_index == 3


def test_per_node_rdma_route_overrides_homogeneous_defaults():
    config = MooncakeConfig(
        local_hostname="10.0.0.2",
        protocol="rdma",
        device_name="mlx5_default",
        rdma_port=1,
        rdma_gid_index=3,
        rdma_routes={
            "10.0.0.2": {
                "device_name": "mlx5_7",
                "port": 2,
                "gid_index": 5,
            }
        },
    )

    assert config.device_name == "mlx5_7"
    assert config.rdma_port == 2
    assert config.rdma_gid_index == 5


def test_partial_per_node_rdma_route_preserves_homogeneous_defaults():
    config = MooncakeConfig(
        local_hostname="10.0.0.2",
        protocol="rdma",
        device_name="mlx5_default",
        rdma_port=1,
        rdma_gid_index=3,
        rdma_routes={"10.0.0.2": {"gid_index": 5}},
    )

    assert config.device_name == "mlx5_default"
    assert config.rdma_port == 1
    assert config.rdma_gid_index == 5


@pytest.mark.parametrize(
    "routes,match",
    [
        ({"10.0.0.1": []}, "must be a mapping"),
        ({"10.0.0.1": {"gid_indx": 3}}, "unknown fields"),
        ({"10.0.0.1": {"port": True}}, "port >= 1"),
    ],
)
def test_rdma_route_malformed_overrides_fail_closed(routes, match):
    with pytest.raises(ValueError, match=match):
        resolve_rdma_routes(
            ["10.0.0.1"],
            device_name="mlx5_0",
            port=1,
            gid_index=3,
            overrides=routes,
        )


def test_transport_native_environment_does_not_leak_across_config_switches():
    rdma = MooncakeConfig(
        protocol="rdma",
        device_name="mlx5_0",
        rdma_port=1,
        rdma_gid_index=3,
        data_plane="transfer_engine",
        enable_gpu_direct=True,
        allow_host_fallback=False,
    )
    tcp = MooncakeConfig(protocol="tcp")
    nvlink = MooncakeConfig(protocol="nvlink_intra")

    with mock.patch.dict(os.environ, {}, clear=True):
        rdma.apply_env_defaults()
        assert os.environ["MC_FORCE_HCA"] == "1"
        assert os.environ["MC_TRANSFER_TIMEOUT"] == "300"
        assert "MC_FORCE_TCP" not in os.environ
        assert "MC_TCP_PROGRESS_TIMEOUT_SEC" not in os.environ

        nvlink.apply_env_defaults()
        assert "MC_FORCE_HCA" not in os.environ
        assert "MC_FORCE_TCP" not in os.environ
        assert "MC_TCP_PROGRESS_TIMEOUT_SEC" not in os.environ
        assert os.environ["MC_INTRA_NVLINK"] == "1"

        tcp.apply_env_defaults()
        assert "MC_FORCE_HCA" not in os.environ
        assert os.environ["MC_FORCE_TCP"] == "1"
        assert os.environ["MC_TCP_PROGRESS_TIMEOUT_SEC"] == "300"
        assert "MC_INTRANODE_NVLINK" not in os.environ
        assert "MC_INTRA_NVLINK" not in os.environ
        assert "MC_USE_NVLINK_IPC" not in os.environ

        rdma.apply_env_defaults()
        assert os.environ["MC_FORCE_HCA"] == "1"
        assert "MC_FORCE_TCP" not in os.environ
        assert "MC_TCP_PROGRESS_TIMEOUT_SEC" not in os.environ


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"enable_gpu_direct": False}, "enable_gpu_direct"),
        ({"allow_host_fallback": True}, "allow_host_fallback"),
        (
            {
                "data_plane": "store",
                "protocol": "tcp",
                "require_nvlink": True,
            },
            "require_nvlink",
        ),
    ],
)
def test_transfer_engine_invalid_combinations_fail_closed(overrides, match):
    values = {
        "data_plane": "transfer_engine",
        "protocol": "nvlink_intra",
        "enable_gpu_direct": True,
        "allow_host_fallback": False,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        MooncakeConfig(**values)


@pytest.mark.parametrize("length", [1, 9, 2048, 2049])
@pytest.mark.parametrize("cp_size", [1, 3, 8])
@pytest.mark.parametrize("include_last", [False, True])
def test_context_shard_buffer_covers_uneven_floor_partitions(length, cp_size, include_last):
    from aspec.transfer.mooncake.helpers import calculate_eagle3_context_shard_buffer_size

    hidden_dim, aux_layers = 4096, 3
    capacity = calculate_eagle3_context_shard_buffer_size(
        length, hidden_dim, cp_size, aux_layers, include_last
    )
    payloads = []
    for rank in range(cp_size):
        local_length = length * (rank + 1) // cp_size - length * rank // cp_size
        payloads.append(local_length * hidden_dim * 2 * (aux_layers + int(include_last)) + length * 8)
    assert capacity == (max(payloads) + 255) // 256 * 256


@pytest.mark.parametrize(
    "explicit_size,expected_size",
    [(None, 6_444_548_096), (12_886_999_040, 12_886_999_040)],
)
def test_direct_cp_slot_auto_size_preserves_explicit_override(explicit_size, expected_size):
    args = SimpleNamespace(
        mooncake_local_hostname="worker",
        mooncake_data_plane="transfer_engine",
        mooncake_protocol="nvlink_intra",
        mooncake_enable_gpu_direct=True,
        mooncake_allow_host_fallback=False,
        mooncake_max_seq_len=262144,
        mooncake_hidden_dim=4096,
        mooncake_gpu_buffer_size=explicit_size,
        mooncake_source_gpu_ring_slots=6,
        mooncake_destination_gpu_ring_slots=2,
        aux_hidden_states_layers=[1, 9, 17, 25, 33],
        attention_backend="fa4",
        draft_context_parallel_size=2,
    )
    config = MooncakeConfig.from_flat_args(args)
    assert config.gpu_buffer_size == expected_size
    assert config.ring_slots_for_role("source") == 6
    assert config.ring_slots_for_role("destination") == 2
