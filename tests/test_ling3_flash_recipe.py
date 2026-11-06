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

import json
from pathlib import Path

from aspec.config.train_config import load_config
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.planning import _resolve_target_family, compile_execution_plan
from tools.convert_to_hf import _serving_config

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "aspec/config/dspark_ling3_flash_draft_config.json"


def test_flash_recipe_resolves_without_downloading_or_allocating_gpus():
    config = load_config(str(ROOT / "configs/recipes/ling3_flash_dspark.yaml"))
    config.model.draft_model_config = str(CONFIG)
    plan = compile_execution_plan(config).to_dict()
    assert plan["workload"]["models"]["target"]["family"] == "ling3"
    features = plan["workload"]["features"]
    assert features["target_hidden_size"] == 2560
    assert features["target_num_hidden_layers"] == 42
    assert features["aux_hidden_state_layers"] == [1, 11, 23, 29, 35]
    assert features["tensors"]["hidden_states"]["shape"] == ["sequence", 12800]
    assert features["tensors"]["last_hidden_states"]["value_stage"] == "post_norm"
    assert config.training.dflash_block_size == 8
    assert config.training.fsdp_strategy == "REPLICATE"
    assert config.inference.sglang.tp_size == 2
    assert config.inference.target_prefill_memory_optimized is False
    assert config.mooncake.require_nvlink is True


def test_flash_draft_matches_official_dspark_structure_and_export_contract():
    import torch

    raw = json.loads(CONFIG.read_text())
    config = AutoDraftModelConfig.from_file(str(CONFIG))
    with torch.device("meta"):
        model = AutoEagle3DraftModel.from_config(config)
    assert config.hidden_size == 2560
    assert config.num_hidden_layers == 5
    assert config.head_dim == 128
    assert config.num_attention_heads == config.num_key_value_heads == 32
    # Official Flash DSpark uses PAD as mask, unlike the tiny recipe's gMASK.
    assert config.mask_token_id == 156892
    assert model.context_proj.weight.shape == (2560, 12800)
    assert model.layers[0].self_attn.q_proj.weight.shape == (4096, 2560)
    assert model.layers[0].self_attn.k_proj.weight.shape == (4096, 2560)
    exported = _serving_config(raw)
    assert exported["model_type"] == "qwen3"
    assert exported["num_target_layers"] == 5
    assert exported["dflash_config"] == {
        "num_target_layers": 42, "target_layer_ids": [1, 11, 23, 29, 35],
    }
    assert exported["block_size"] == 8
    assert _resolve_target_family("inclusionAI/Ling-3.0-flash") == "ling3"
