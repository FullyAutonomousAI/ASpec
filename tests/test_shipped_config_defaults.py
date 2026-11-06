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

from pathlib import Path

from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).parents[1]


def test_shipped_small_draft_configs_use_replicate():
    """Do not revive historical FULL_SHARD benchmark settings as defaults."""

    config_paths = sorted(
        path
        for path in (_repo_root() / "configs/recipes").glob("*.yaml")
        if not path.name.startswith("._")
    )
    assert config_paths
    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        assert config.training.fsdp_strategy == "REPLICATE", config_path.name


def test_primary_multinode_reference_uses_current_natural_path():
    config = OmegaConf.load(
        _repo_root() / "configs/recipes/qwen3_8b_4k_multinode_node_local_nvlink.yaml"
    )

    assert config.training.attention_backend == "cudnn_flex_attention"
    assert config.training.fsdp_strategy == "REPLICATE"
    assert config.training.training_num_nodes == 2
    assert config.training.training_num_gpus_per_node == 4
    assert config.inference.inference_num_gpus == 8
    assert config.inference.inference_num_gpus_per_node == 4
    assert config.mooncake.feature_affinity == "node_local_required"
    assert config.mooncake.protocol == "nvlink_intra"
    assert config.mooncake.require_nvlink is True


def test_role_separated_control_uses_full_target_and_draft_replicas():
    config = OmegaConf.load(
        _repo_root() / "configs/examples/qwen3_8b_4k_multinode_tcp.yaml"
    )

    assert config.training.fsdp_strategy == "REPLICATE"
    assert config.training.training_num_gpus_per_node == 8
    assert config.inference.inference_num_gpus == 8
    assert len(config.inference.inference_node_ips) == 1
