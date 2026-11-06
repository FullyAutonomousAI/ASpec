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

"""Public algorithm boundaries; no GPU kernels or training compilation."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from aspec.config.train_config import load_config
from aspec.feature_contract import build_target_feature_contract
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.multi_draft import compile_multi_draft_plan
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.training.eagle3_trainer import Eagle3Trainer
from aspec.training.trainer_actor import _trainer_class_for_config
from tools.convert_to_hf import _draft_structure_manifest, _serving_config

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2", "eagle3"])
def test_public_algorithm_recipe_compiles_and_reports_real_algorithm(algorithm):
    config = load_config(str(ROOT / f"configs/{'examples' if algorithm == 'dflash2' else 'recipes'}/qwen3_8b_{algorithm}.yaml"), save_snapshot=False)
    plan = compile_execution_plan(config)
    assert plan.workload.objective["family"] == algorithm
    assert plan.workload.models["draft"]["algorithm"] == algorithm
    assert plan.to_runtime_args().draft_algorithm == algorithm
    if algorithm == "eagle3":
        assert "block_size" not in plan.workload.objective
        assert plan.policy.pipeline["eagle3_target_cache"] == "logits"
        assert plan.policy.pipeline["eagle3_loss_gradient_checkpointing"] is False


def _shared_algorithm_config():
    config = load_config(
        str(ROOT / "configs/examples/qwen3_8b_multi_algorithm_feature_views.yaml"),
        save_snapshot=False,
    )
    config.experiments[1].model.draft_model_config = str(
        ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json"
    )
    config.experiments[2].model.draft_model_config = str(ROOT / "aspec/config/dflash_draft_config.json")
    config.experiments[2].training.dflash_num_target_layers = 5
    return config


def test_distinct_algorithms_share_identical_payloads_without_losing_algorithm_identity():
    config = _shared_algorithm_config()
    plan = compile_multi_draft_plan(config)
    assert {p.workload.objective["family"] for p in plan.plans.values()} == {
        "dspark", "dflash", "eagle3",
    }
    assert len({p.workload.features["contract_id"] for p in plan.plans.values()}) == 1
    assert len({p.plan_id for p in plan.plans.values()}) == 3


def test_shared_payload_contract_still_distinguishes_feature_content():
    values = dict(target_hidden_size=64, target_num_hidden_layers=12,
                  aux_hidden_state_layers=[1, 5, 9], store_last_hidden_states=True)
    dspark = build_target_feature_contract(drafter_family="dspark", **values)
    eagle = build_target_feature_contract(drafter_family="eagle3", **values)
    assert dspark == eagle
    other = build_target_feature_contract(drafter_family="eagle3", **{
        **values, "aux_hidden_state_layers": [2, 5, 9],
    })
    assert other["contract_id"] != dspark["contract_id"]


@pytest.mark.parametrize("change,match", [
    ({"dflash_packing": True}, "packing"),
    ({"eagle3_num_anchors": 4}, "anchored"),
    ({"attention_backend": "cudnn_flex_attention"}, "attention backend"),
    ({"compile_model": True}, "whole-model compile"),
])
def test_eagle3_unsupported_execution_fails_before_actor_creation(change, match):
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    for key, value in change.items():
        config.training[key] = value
    with pytest.raises((PlanCompileError, ValueError), match=match):
        compile_execution_plan(config)


@pytest.mark.parametrize("change,match", [
    ({"eagle3_target_cache": "probabilities"}, "must be 'lazy' or 'logits'"),
    ({"eagle3_target_cache": "logits", "attention_backend": "fa4"}, "logits target cache"),
])
def test_eagle3_target_cache_contract_fails_before_actor_creation(change, match):
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    for key, value in change.items():
        config.training[key] = value
    with pytest.raises((PlanCompileError, ValueError), match=match):
        compile_execution_plan(config)


def test_eagle3_pruned_vocab_requires_lazy_target_cache():
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    config.model.draft_model_config = str(
        ROOT / "aspec/config/eagle3_qwen3_8b_pruned_draft_config.json"
    )
    with pytest.raises(PlanCompileError, match="vocabulary pruning"):
        compile_execution_plan(config)
    config.training.eagle3_target_cache = "lazy"
    config.training.eagle3_loss_gradient_checkpointing = True
    assert compile_execution_plan(config).policy.pipeline["eagle3_target_cache"] == "lazy"


@pytest.mark.parametrize("checkpoint", [False, True])
def test_eagle3_plan_resolves_inherited_loss_checkpoint_policy(checkpoint):
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    config.training.eagle3_loss_gradient_checkpointing = None
    config.training.gradient_checkpointing = checkpoint
    plan = compile_execution_plan(config)
    assert plan.policy.pipeline["eagle3_loss_gradient_checkpointing"] is checkpoint


def test_eagle3_model_and_trainer_dispatch_and_export_identity():
    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    raw.update(hidden_size=32, target_hidden_size=32, intermediate_size=64,
               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               vocab_size=64, draft_vocab_size=64, pad_token_id=0)
    config = AutoDraftModelConfig.from_dict(raw)
    assert _trainer_class_for_config(config) is Eagle3Trainer
    model = AutoEagle3DraftModel.from_config(config)
    assert model.fc.in_features == 32 * 5
    serving = _serving_config(raw)
    assert serving["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == raw["target_layer_ids"]
    assert "dflash_config" not in serving
    assert _draft_structure_manifest(raw)["loader_contract"]["speculative_algorithm"] == "EAGLE3"


def test_dflash_export_does_not_claim_dspark():
    raw = json.loads((ROOT / "aspec/config/dflash_draft_config.json").read_text())
    assert _draft_structure_manifest(raw)["loader_contract"]["speculative_algorithm"] == "DFLASH"


def test_exact_resume_rejects_algorithm_or_eagle3_objective_change():
    from argparse import Namespace

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    args = Namespace(world_size=1, draft_algorithm="eagle3", ttt_length=3,
                     loss_type="forward_kl", lk_eta=3.0, ploss_weights=None)
    saved = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    args.ttt_length = 4
    with pytest.raises(RuntimeError, match="training_objective"):
        _validate_exact_resume_topology(args, saved)
    args.ttt_length = 3
    args.draft_algorithm = "dflash"
    args.dflash_block_size = 7
    with pytest.raises(RuntimeError, match="draft_algorithm"):
        _validate_exact_resume_topology(args, saved)


def test_unknown_export_algorithm_does_not_default_to_dspark():
    raw = {"architectures": ["UnknownDraft"], "hidden_size": 64}
    with pytest.raises(ValueError, match="Unsupported Draft algorithm"):
        _draft_structure_manifest(raw)


def test_shared_source_builds_union_for_different_layer_content(tmp_path):
    config = _shared_algorithm_config()
    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    raw["target_layer_ids"][0] = 2
    path = tmp_path / "different-eagle3.json"
    path.write_text(json.dumps(raw))
    config.experiments[1].model.draft_model_config = str(path)
    plan = compile_multi_draft_plan(config)
    assert plan.source_plan.workload.features["aux_hidden_state_layers"] == [1, 2, 9, 17, 25, 33]
    assert plan.plans["eagle3_cp2"].workload.features["aux_hidden_state_layers"] == [2, 9, 17, 25, 33]
    from aspec.config.mooncake_config import MooncakeConfig
    assert MooncakeConfig._infer_num_aux_layers(plan.to_runtime_args()) == 6


@pytest.mark.parametrize("change", [
    {"ttt_length": 0}, {"loss_type": "wrong"}, {"lk_eta": float("nan")},
    {"ploss_weights": [0.0, 0.0, 0.0]}, {"ploss_weights": [1.0]},
])
def test_invalid_eagle3_objective_fails_before_startup(change):
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    for key, value in change.items():
        config.training[key] = value
    with pytest.raises(ValueError, match="EAGLE3"):
        compile_execution_plan(config)


def test_export_uses_real_training_block_size_and_fences_algorithm(tmp_path):
    from tools.convert_to_hf import _checkpoint_training_contract

    model_dir = tmp_path / "checkpoints" / "iter_0000017" / "model"
    model_dir.mkdir(parents=True)
    (tmp_path / "config.yaml").write_text("training:\n  dflash_block_size: 7\n")
    raw = {"architectures": ["DFlashDraftModel"]}
    assert _checkpoint_training_contract(model_dir, raw) == {"algorithm": "dflash", "block_size": 7}
    with pytest.raises(ValueError, match="block_size conflicts"):
        _checkpoint_training_contract(model_dir, {**raw, "block_size": 16})
    (model_dir.parent / "meta.json").write_text(json.dumps({
        "training_topology": {"draft_algorithm": "eagle3"},
    }))
    with pytest.raises(ValueError, match="algorithm conflicts"):
        _checkpoint_training_contract(model_dir, raw)


def test_relocated_dflash_checkpoint_retains_training_block_size(tmp_path):
    from tools.convert_to_hf import _checkpoint_training_contract

    model = tmp_path / "iter_0000017" / "model"
    model.mkdir(parents=True)
    (model.parent / "meta.json").write_text(json.dumps({
        "training_topology": {"draft_algorithm": "dflash", "training_objective": {"block_size": 7}},
    }))
    assert _checkpoint_training_contract(model, {"architectures": ["DFlashDraftModel"]})["block_size"] == 7


def test_exact_resume_rejects_dflash_loss_change_but_keeps_legacy_contract():
    from argparse import Namespace

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    args = Namespace(world_size=1, draft_algorithm="dflash", dflash_block_size=7,
                     dflash_ce_loss_alpha=1.0, dflash_l1_loss_alpha=0.0)
    saved = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    args.dflash_l1_loss_alpha = 0.9
    with pytest.raises(RuntimeError, match="training_objective"):
        _validate_exact_resume_topology(args, saved)
    saved["training_topology"]["training_objective"] = {"block_size": 7}
    _validate_exact_resume_topology(args, saved)


def test_same_shape_semantic_changes_cannot_resume_or_export(tmp_path):
    from argparse import Namespace

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )
    from tools.convert_to_hf import _checkpoint_training_contract, _resolve_config_path

    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    args = Namespace(world_size=1, draft_algorithm="eagle3",
                     draft_model_config_obj=AutoDraftModelConfig.from_dict(raw))
    saved = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    raw["rope_theta"] *= 2
    args.draft_model_config_obj = AutoDraftModelConfig.from_dict(raw)
    with pytest.raises(RuntimeError, match="draft_model_semantics"):
        _validate_exact_resume_topology(args, saved)
    model = tmp_path / "iter_0000017" / "model"
    model.mkdir(parents=True)
    (model.parent / "meta.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="model semantics"):
        _checkpoint_training_contract(model, raw)
    saved_config = model.parent / "draft_config.json"
    saved_config.write_text(json.dumps(raw))
    assert _resolve_config_path(model.parent, None) == saved_config


def test_eagle3_fa4_thd_cp_is_not_constrained_by_unused_dspark_anchors():
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False,
                         cli_args=["training.attention_backend=fa4",
                                   "training.eagle3_target_cache=lazy",
                                   "training.eagle3_loss_gradient_checkpointing=true",
                                   "training.draft_context_parallel_size=3",
                                   "training.training_num_gpus_per_node=3",
                                   "training.dflash_packing=true"])
    plan = compile_execution_plan(config)
    assert plan.to_runtime_args().draft_context_parallel_size == 3
    assert plan.workload.objective["family"] == "eagle3"


def test_dflash2_nested_config_and_dispatch(tmp_path):
    from aspec.training.dflash2_trainer import DFlash2Trainer
    from tools.convert_to_hf import _checkpoint_training_contract

    raw = json.loads((ROOT / "aspec/config/dflash2_draft_config.json").read_text())
    raw["dflash_config"] = {"block_size": raw.pop("block_size"),
                            "target_layer_ids": raw.pop("target_layer_ids")}
    raw["num_target_layers"] = raw.pop("target_num_hidden_layers")
    config = AutoDraftModelConfig.from_dict(raw)
    assert config.num_target_layers == 5
    assert config.target_num_hidden_layers == 36
    assert _trainer_class_for_config(config) is DFlash2Trainer
    serving = _serving_config(config.to_dict())
    assert serving["dflash_config"]["block_size"] == 8
    assert serving["num_target_layers"] == 36
    assert _draft_structure_manifest(config.to_dict())["loader_contract"]["speculative_algorithm"] == "DFLASH"
    model = tmp_path / "model"
    model.mkdir()
    assert _checkpoint_training_contract(model, raw)["block_size"] == 8


def test_dflash2_plan_preserves_algorithm_specific_objective_and_cp():
    config = load_config(str(ROOT / "configs/examples/qwen3_8b_dflash2.yaml"), save_snapshot=False)
    config.training.attention_backend = "fa4"
    config.training.draft_context_parallel_size = 2
    config.training.dflash_packing = True
    plan = compile_execution_plan(config)
    assert plan.workload.objective["selector_objective"] == "experimental_teacher_forced_topk_ce"
    assert plan.to_runtime_args().draft_context_parallel_size == 2
    config.training.dflash_block_size = 7
    with pytest.raises(ValueError, match="convolution block_size"):
        compile_execution_plan(config)


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_dflash2_invalid_selector_weight_fails_preflight(alpha):
    config = load_config(str(ROOT / "configs/examples/qwen3_8b_dflash2.yaml"), save_snapshot=False)
    config.training.dflash2_selector_loss_alpha = alpha
    with pytest.raises(ValueError, match="selector_loss_alpha"):
        compile_execution_plan(config)


def test_model_semantics_accept_equivalent_defaults_and_relocated_paths():
    from aspec.config.draft_algorithm import draft_model_semantics

    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    base = draft_model_semantics(AutoDraftModelConfig.from_dict(raw))
    assert draft_model_semantics(base) == base
    raw.update(_name_or_path="/different/machine/draft", transformers_version="future",
               fc_norm=False, norm_output=False, lm_head_rank_divisor=0)
    assert draft_model_semantics(AutoDraftModelConfig.from_dict(raw)) == base


@pytest.mark.parametrize("change,match", [
    ({"lm_head_rank_divisor": 2}, "factorized lm_head"),
    ({"target_layer_ids": [1], "num_aux_hidden_states": 1}, "skip the trained FC"),
    ({"use_aux_norm": True, "fc_norm": False}, "serving-only normalization"),
])
def test_eagle3_export_cannot_silently_change_trained_computation(change, match):
    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _serving_config({**raw, **change})


@pytest.mark.parametrize("mla", [False, True])
def test_eagle_runtime_attention_policy_matches_plan_for_both_projections(mla):
    from aspec.config.draft_attention import resolve_draft_attention

    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    if mla:
        config.model.draft_model_config = str(ROOT / "aspec/config/eagle3_qwen3_8b_mla_draft_config.json")
        config.training.attention_backend = "fa4"
        config.training.eagle3_target_cache = "lazy"
        config.training.eagle3_loss_gradient_checkpointing = True
    plan = compile_execution_plan(config)
    runtime = plan.to_runtime_args()
    loaded = AutoDraftModelConfig.from_dict(json.loads(Path(runtime.draft_model_config).read_text()))
    observed = resolve_draft_attention(loaded.to_dict())
    planned = plan.workload.models["draft"]["attention"]
    assert observed["window_semantics"] == planned["window_semantics"] == "causal_ttt"
    assert observed["draft_block_attention"] == planned["draft_block_attention"] == "causal_ttt_cache"

@pytest.mark.parametrize("field,value", [("num_experts_per_tok", 1), ("norm_topk_prob", False),
                                          ("router_aux_loss_coef", 0.1)])
@pytest.mark.parametrize("legacy", [False, True])
def test_moe_routing_resume_and_export_are_bound_to_saved_policy(tmp_path, field, value, legacy):
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )
    from tools.convert_to_hf import _checkpoint_training_contract
    raw = json.loads((ROOT / "aspec/config/dspark_draft_config.json").read_text())
    raw.update(ffn_type="qwen3_moe", num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
               norm_topk_prob=True, router_aux_loss_coef=0.001)
    config = AutoDraftModelConfig.from_dict(raw)
    args = Namespace(world_size=1, draft_algorithm="dspark", dflash_block_size=4,
                     draft_model_config_obj=config)
    topology = _logical_training_topology(args, 1)
    if legacy:
        topology.pop("draft_ffn_policy")
    metadata = {"world_size": 1, "training_topology": topology}
    model = tmp_path / "iter_0000001" / "model"
    model.mkdir(parents=True)
    (model.parent / "draft_config.json").write_text(json.dumps(raw))
    (model.parent / "meta.json").write_text(json.dumps(metadata))
    _validate_exact_resume_topology(args, metadata, checkpoint_dir=model.parent)
    _checkpoint_training_contract(model, raw)
    changed = {**raw, field: value}
    args.draft_model_config_obj = AutoDraftModelConfig.from_dict(changed)
    with pytest.raises(RuntimeError, match="FFN/routing"):
        _validate_exact_resume_topology(args, metadata, checkpoint_dir=model.parent)
    with pytest.raises(ValueError, match="FFN/routing"):
        _checkpoint_training_contract(model, changed)


def test_legacy_moe_without_saved_configuration_fails_closed(tmp_path):
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )
    raw = json.loads((ROOT / "aspec/config/dspark_draft_config.json").read_text())
    raw.update(ffn_type="qwen3_moe", num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    args = Namespace(world_size=1, draft_algorithm="dspark", dflash_block_size=4,
                     draft_model_config_obj=AutoDraftModelConfig.from_dict(raw))
    topology = _logical_training_topology(args, 1)
    topology.pop("draft_ffn_policy")
    with pytest.raises(RuntimeError, match="cannot verify legacy MoE"):
        _validate_exact_resume_topology(args, {"world_size":1,"training_topology":topology}, checkpoint_dir=tmp_path)
