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

"""The public three-structure recipe uses the normal CLI, not diagnostic hooks."""

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from aspec.models.draft import AutoDraftModelConfig
from aspec.multi_draft import compile_multi_draft_plan

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/examples/ling3_flash_multi_structure_dspark.yaml"
STRUCTURES = {"dense": "dense", "qwen_moe": "qwen3_moe", "ling_moe": "ling_moe"}


def _config(monkeypatch):
    monkeypatch.chdir(ROOT)
    return load_config(str(RECIPE), save_snapshot=False)


def test_public_recipe_shares_target_and_keeps_comparison_controls_equal(monkeypatch):
    config = _config(monkeypatch)
    plan = compile_multi_draft_plan(config)
    assert plan.to_dict()["resources"] == {
        "target_gpus": 2, "draft_gpus": {"dense": 1, "qwen_moe": 1, "ling_moe": 1},
    }
    assert plan.source_seed == 42
    contracts = [child.workload.features for child in plan.plans.values()]
    assert contracts[0] == contracts[1] == contracts[2]
    assert contracts[0]["aux_hidden_state_layers"] == [1, 11, 23, 29, 35]
    for name, child in plan.plans.items():
        args = child.to_runtime_args()
        assert child.workload.models["draft"]["structure"]["ffn_type"] == STRUCTURES[name]
        assert child.workload.models["target"]["family"] == "ling3"
        assert args.global_batch_size == 4
        assert args.seed == args.dataset_seed == 42
        assert args.num_train_steps == 16 and args.lr_total_steps == 32
        assert args.learning_rate == 6e-4 and args.min_lr == 6e-5
        assert args.fsdp_strategy == "REPLICATE"
        assert args.draft_context_parallel_size == 1 and not args.dflash_packing
        assert args.mooncake_require_nvlink and not args.mooncake_allow_host_fallback
        assert not args.continual_training and args.load_path is None
        assert args.save_final_checkpoint and args.save_interval == 8
        assert Path(args.output_dir).name == name
        model_path = Path(args.draft_model_config)
        assert model_path.is_file() and model_path.parent == ROOT / "aspec/config"
        parsed = AutoDraftModelConfig.from_file(str(model_path))
        assert parsed.ffn_type == STRUCTURES[name]


def test_promoted_qwen_config_preserves_validated_structure():
    official = json.loads((ROOT / "aspec/config/dspark_ling3_flash_draft_config.json").read_text())
    qwen = json.loads((ROOT / "aspec/config/dspark_ling3_flash_qwen_moe_draft_config.json").read_text())
    assert {key: qwen[key] for key in official} == official
    assert qwen["num_experts"] == 8 and qwen["num_experts_per_tok"] == 2
    assert qwen["moe_intermediate_size"] == 768
    assert qwen["router_aux_loss_coef"] == 0.001


def test_public_recipe_dry_run_uses_cli_without_starting_ray(monkeypatch, capsys, tmp_path):
    from aspec import train_entry

    _config(monkeypatch)
    output = tmp_path / "unused-output"
    monkeypatch.setattr(
        sys, "argv", ["aspec", "--config", str(RECIPE), "--dry-run", f"output_dir={output}"],
    )
    monkeypatch.setattr(train_entry.ray, "init", lambda **_: pytest.fail("dry-run started Ray"))
    with pytest.raises(SystemExit) as stopped:
        train_entry._prepare_config()
    assert stopped.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["experiments"]) == set(STRUCTURES)
    assert not output.exists()


def test_public_resume_configuration_preserves_structure_schedule_and_source(monkeypatch, tmp_path):
    config = _config(monkeypatch)
    original = compile_multi_draft_plan(config)
    resumed = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    resumed.output_dir = str(tmp_path / "resume")
    resumed.training.num_train_steps = 18
    for experiment in resumed.experiments:
        experiment.training.load_path = str(
            Path(config.output_dir) / experiment.name / "checkpoints"
        )
    plan = compile_multi_draft_plan(load_config(base_config=resumed, save_snapshot=False))
    assert plan.source_seed == original.source_seed
    for name, child in plan.plans.items():
        args = child.to_runtime_args()
        assert args.num_train_steps == 18 and args.lr_total_steps == 32
        assert not args.continual_training
        assert args.load_path == str(Path(config.output_dir) / name / "checkpoints")
        assert args.output_dir == str(tmp_path / "resume" / name)
        assert args.global_batch_size == 4
        assert child.workload.features == original.plans[name].workload.features


@pytest.mark.parametrize("change", ["packing", "context"])
def test_recipe_accepts_supported_shared_source_layouts(monkeypatch, change):
    config = _config(monkeypatch)
    if change == "packing":
        config.training.dflash_packing = True
    elif change == "context":
        config.training.attention_backend = "fa4"
        config.training.training_num_gpus_per_node = 2
        config.training.draft_context_parallel_size = 2
    plan = compile_multi_draft_plan(config)
    for child in plan.plans.values():
        args = child.to_runtime_args()
        assert args.dflash_packing == (change == "packing")
        assert args.draft_context_parallel_size == (2 if change == "context" else 1)


def test_recipe_rejects_incompatible_shared_feature_contract(monkeypatch):
    config = _config(monkeypatch)
    config.experiments[1].model = {
        "draft_model_config": str(ROOT / "aspec/config/dspark_qwen3_8b_moe_draft_config.json")
    }
    with pytest.raises(ValueError, match="feature"):
        compile_multi_draft_plan(config)
