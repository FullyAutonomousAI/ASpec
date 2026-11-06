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

import pytest
import torch

from aspec.config.train_config import load_config
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.dspark import K3DSparkMLAAttention
from aspec.models.draft.ling_moe import LingDraftMoE
from aspec.planning import _resolve_target_family, compile_execution_plan
from tools.convert_to_hf import _serving_config

ROOT = Path(__file__).parents[1]
DRAFT = ROOT / "aspec/config/dspark_ling31_flash_mla_moe_draft_config.json"


def test_ling31_remote_name_and_real_mla_moe_model_contract(monkeypatch):
    monkeypatch.setenv("ASPEC_TRAIN_DATA", "/tmp/ling31/documents.parquet")
    config = load_config(str(ROOT / "configs/recipes/ling31_flash_mla_moe_dspark.yaml"))
    config.model.draft_model_config = str(DRAFT)
    plan = compile_execution_plan(config).to_dict()
    assert _resolve_target_family("inclusionAI/Ling-3.1-flash") == "ling3"
    assert plan["workload"]["features"]["target_hidden_size"] == 4096
    assert plan["workload"]["features"]["aux_hidden_state_layers"] == [7, 23, 39, 55, 71]
    assert config.training.dflash_packing
    assert config.training.fsdp_strategy == "REPLICATE"
    assert config.training.streaming_dispatch_with_training
    assert not config.training.pipeline_dispatch_with_training
    assert config.inference.target_prefill_memory_optimized is False
    raw = json.loads(DRAFT.read_text())
    with torch.device("meta"):
        model = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_file(str(DRAFT)))
    for layer in model.layers:
        assert isinstance(layer.self_attn, K3DSparkMLAAttention)
        assert isinstance(layer.mlp, LingDraftMoE)
        assert layer.mlp.gate.expert_bias.dtype == torch.float32
    assert model.context_proj.weight.shape == (4096, 20480)
    assert model.layers[0].self_attn.q_b_proj.weight.shape == (6144, 1536)
    assert model.layers[0].mlp.shared_experts.gate_proj.weight.shape == (1280, 4096)
    assert raw["mask_token_id"] == 156892
    with pytest.raises(NotImplementedError, match="--format aspec"):
        _serving_config(raw)
