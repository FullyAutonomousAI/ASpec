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

"""Small deterministic structure/objective tests; no GPU or kernel compilation."""

import copy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from aspec.config.draft_structure import resolve_draft_structure
from aspec.config.train_config import load_config
from aspec.models.dflash import DFlashModel
from aspec.models.draft.auto import AutoDraftModelConfig
from aspec.models.draft.dflash import DFlashAttention, DFlashConfig, DFlashDraftModel
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.draft.moe import QwenDraftMoE, router_loss_terms
from aspec.models.dspark import DSparkModel
from aspec.multi_draft import compile_multi_draft_plan
from aspec.planning import PlanCompileError, compile_execution_plan
from tools.convert_to_hf import _serving_config

ROOT = Path(__file__).parents[1]
MOE = dict(ffn_type="qwen3_moe", num_experts=4, num_experts_per_tok=2,
           moe_intermediate_size=12, norm_topk_prob=True, router_aux_loss_coef=0.01)


def draft_config(cls=DSparkConfig, **overrides):
    settings = {**MOE, **overrides}
    if settings["ffn_type"] == "dense":
        settings = {"ffn_type": "dense", "num_experts": 0}
    return cls(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
               num_attention_heads=2, num_key_value_heads=1, vocab_size=31,
               max_position_embeddings=64, num_target_layers=2,
               target_hidden_size=16, target_num_hidden_layers=12,
               target_layer_ids=[1, 9], mask_token_id=30, markov_rank=4,
               **settings)


@pytest.mark.parametrize("change", [
    {"ffn_type": "unknown"}, {"ffn_type": "dense"},
    {"num_experts": 0}, {"num_experts": True}, {"num_experts_per_tok": 5},
    {"num_experts_per_tok": 1.5}, {"moe_intermediate_size": 0},
    {"router_aux_loss_coef": -1}, {"router_aux_loss_coef": float("nan")},
    {"router_aux_loss_coef": float("inf")}, {"norm_topk_prob": "false"},
    {"num_shared_experts": 1}, {"n_group": 8}, {"score_function": "sigmoid"},
    {"moe_router_enable_expert_bias": True}, {"routed_scaling_factor": 2.5},
    {"mlp_only_layers": [0]}, {"decoder_sparse_step": 2},
    {"hidden_act": "gelu"}, {"topk_method": "noaux_tc"},
])
def test_invalid_structure_rejected_before_model_allocation(change):
    with pytest.raises(ValueError):
        resolve_draft_structure({**MOE, **change})


def test_dense_default_has_unchanged_parameter_names_and_values():
    torch.manual_seed(4)
    default = DFlashDraftModel(draft_config(DFlashConfig, ffn_type="dense", num_experts=0))
    torch.manual_seed(4)
    explicit = DFlashDraftModel(draft_config(DFlashConfig, ffn_type="dense", num_experts=0))
    assert all("experts" not in key and ".gate.weight" not in key for key in default.state_dict())
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("norm", [False, True])
def test_qwen_moe_matches_official_outputs_and_gradients(norm):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    torch.manual_seed(5)
    config = draft_config(norm_topk_prob=norm)
    model = QwenDraftMoE(config)
    ref = Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(
        hidden_size=16, intermediate_size=32, moe_intermediate_size=12,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=norm,
    ))
    with torch.no_grad():
        ref.gate.weight.copy_(model.gate.weight)
        packed = hasattr(ref.experts, "gate_up_proj")
        for i, expert in enumerate(model.experts):
            if packed:
                ref.experts.gate_up_proj[i].copy_(torch.cat([
                    expert.gate_proj.weight, expert.up_proj.weight,
                ]))
                ref.experts.down_proj[i].copy_(expert.down_proj.weight)
            else:
                ref.experts[i].load_state_dict(expert.state_dict())
    x = torch.randn(2, 3, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    actual, _ = model(x)
    expected = ref(other)
    if isinstance(expected, tuple):
        expected = expected[0]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    torch.testing.assert_close(model.gate.weight.grad, ref.gate.weight.grad)
    for i, expert in enumerate(model.experts):
        if packed:
            gate_grad, up_grad = ref.experts.gate_up_proj.grad[i].chunk(2)
            down_grad = ref.experts.down_proj.grad[i]
        else:
            gate_grad = ref.experts[i].gate_proj.weight.grad
            up_grad = ref.experts[i].up_proj.weight.grad
            down_grad = ref.experts[i].down_proj.weight.grad
        torch.testing.assert_close(expert.gate_proj.weight.grad, gate_grad)
        torch.testing.assert_close(expert.up_proj.weight.grad, up_grad)
        torch.testing.assert_close(expert.down_proj.weight.grad, down_grad)


def test_unselected_experts_still_have_zero_gradients():
    model = QwenDraftMoE(draft_config(num_experts_per_tok=1))
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.weight[0].fill_(1)
    output, _ = model(torch.ones(1, 3, 16))
    output.sum().backward()
    for i, expert in enumerate(model.experts):
        for parameter in expert.parameters():
            assert parameter.grad is not None
            if i:
                assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("all_masked", [False, True])
def test_auxiliary_matches_qwen_masked_reference_and_ignores_invalid_slots(all_masked):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

    torch.manual_seed(9)
    logits = (torch.randn(6, 4, requires_grad=True), torch.randn(6, 4, requires_grad=True))
    mask = torch.tensor([[1, 1, 0, 1, 0, 0]]) if not all_masked else torch.zeros(1, 6)
    actual, metrics = router_loss_terms(logits, mask, mask.sum(), 2)
    if all_masked:
        assert actual.item() == 0
    else:
        expected = load_balancing_loss_func(logits, 4, 2, attention_mask=mask)
        torch.testing.assert_close(actual / mask.sum(), expected)
    actual.backward()
    for value in logits:
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[mask.flatten() == 0]) == 0
    assert set(metrics) == {"router_aux_loss", "router_max_load", "router_used_experts"}


def _cheap_attention(self, draft_hidden, context_hidden, **kwargs):
    # Isolate FFN/loss plumbing here; GPU integration exercises actual attention.
    return self.o_proj(self.q_proj(draft_hidden)) + context_hidden.mean(1, keepdim=True)


@pytest.mark.parametrize("algorithm", ["dspark", "dflash"])
@pytest.mark.parametrize("all_masked", [False, True])
def test_real_wrapper_trains_router_and_experts_and_accounts_for_auxiliary(algorithm, all_masked):
    torch.manual_seed(13)
    cls, wrapper = (DSparkDraftModel, DSparkModel) if algorithm == "dspark" else (DFlashDraftModel, DFlashModel)
    draft = cls(draft_config(DSparkConfig if algorithm == "dspark" else DFlashConfig))
    draft.freeze_embedding()
    model = wrapper(draft, block_size=3, num_anchors=2)
    batch = dict(input_ids=torch.randint(0, 30, (1, 12)),
                 hidden_states_list=torch.randn(1, 12, 32),
                 loss_mask=torch.zeros(1, 12) if all_masked else torch.ones(1, 12),
                 lm_head_weight=torch.randn(31, 16),
                 last_hidden_states=torch.randn(1, 12, 16),
                 injected_anchors=torch.tensor([[2, 5]]),
                 injected_keep_mask=torch.tensor([[False, False]]) if all_masked else torch.ones(1, 2, dtype=torch.bool))
    with mock.patch.object(DFlashAttention, "forward", _cheap_attention):
        output = model(**batch)
    assert len(output) == 7
    assert "router_aux_loss" in output[5]
    if all_masked:
        assert output[0].item() == 0
    output[0].backward()
    assert draft.embed_tokens.weight.grad is None
    for layer in draft.layers:
        assert layer.mlp.gate.weight.grad is not None
        assert torch.isfinite(layer.mlp.gate.weight.grad).all()
        if not all_masked:
            assert layer.mlp.gate.weight.grad.abs().sum() > 0
        for expert in layer.mlp.experts:
            assert all(parameter.grad is not None for parameter in expert.parameters())
    # Disable only the explicitly separate auxiliary term, not the primary objective.
    draft.config.router_aux_loss_coef = 0
    with mock.patch.object(DFlashAttention, "forward", _cheap_attention):
        without_aux = model(**batch)
    expected_delta = MOE["router_aux_loss_coef"] * output[5]["router_aux_loss"][0]
    torch.testing.assert_close(output[6][0] - without_aux[6][0], expected_delta,
                               atol=2e-5, rtol=1e-4)


def test_plan_expresses_moe_and_shared_source_accepts_dense_plus_moe(tmp_path, monkeypatch):
    base_path = ROOT / "aspec/config/dspark_draft_config.json"
    raw = json.loads(base_path.read_text())
    moe = tmp_path / "moe.json"
    moe.write_text(json.dumps({**raw, **MOE}))
    monkeypatch.setenv("ASPEC_TARGET_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("ASPEC_TRAIN_DATA", str(ROOT / "examples/data/sample_conversations.jsonl"))
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml"))
    config.model.draft_model_config = str(base_path)
    config.experiments[1].model = {"draft_model_config": str(moe)}
    plan = compile_multi_draft_plan(config)
    assert plan.plans["draft_a"].workload.models["draft"]["structure"]["ffn_type"] == "dense"
    assert plan.plans["draft_b"].workload.models["draft"]["structure"]["ffn_type"] == "qwen3_moe"
    assert plan.plans["draft_a"].workload.features == plan.plans["draft_b"].workload.features
    parsed = AutoDraftModelConfig.from_file(str(moe))
    assert parsed.ffn_type == "qwen3_moe"
    assert _serving_config({**raw, **MOE})["ffn_type"] == "qwen3_moe"
    config.experiments = []
    config.model.draft_model_config = str(moe)
    config.training.compile_model = True
    with pytest.raises(PlanCompileError, match="torch.compile"):
        compile_execution_plan(config)


def test_model_state_roundtrip_keeps_every_expert():
    source = DSparkDraftModel(draft_config())
    restored = DSparkDraftModel(DSparkConfig.from_dict(source.config.to_dict()))
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_runtime_observes_experts_and_all_serving_entrypoints_fail_closed(tmp_path):
    from aspec.training.dflash_trainer import DFlashTrainer
    from aspec.training.trainer import Trainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = DSparkDraftModel(draft_config())
    trainer.draft_model.freeze_embedding()
    with mock.patch.object(Trainer, "get_feature_contract_status", return_value={"state": "validated"}):
        observation = trainer.get_feature_contract_status()
    assert observation["draft_structure"]["ffn_modules"] == ["QwenDraftMoE"] * 2
    assert observation["draft_structure"]["expert_counts"] == [4, 4]
    assert observation["draft_structure"]["top_k"] == [2, 2]
    assert observation["draft_structure"]["trainable_parameters"] > 0
    output = tmp_path / "serving"
    with pytest.raises(NotImplementedError, match="MoE Draft serving export"):
        trainer.save_draft_model_for_serving(str(output))
    assert not output.exists()
    trainer.draft_model = DFlashDraftModel(draft_config(DFlashConfig, ffn_type="dense", num_experts=0))
    with mock.patch.object(Trainer, "get_feature_contract_status", return_value={"state": "validated"}):
        assert trainer.get_feature_contract_status() == {"state": "validated"}


def test_dense_config_rejects_silently_ignored_moe_loss():
    with pytest.raises(ValueError, match="explicit ffn_type"):
        DFlashConfig(router_aux_loss_coef=0.01)


def test_moe_recipe_resolves_its_dataset_and_structure_without_gpu():
    config = load_config(str(ROOT / "configs/examples/qwen3_8b_moe_dspark.yaml"))
    assert Path(config.dataset.train_data_path).is_file()
    config.model.draft_model_config = str(ROOT / "aspec/config/dspark_qwen3_8b_moe_draft_config.json")
    plan = compile_execution_plan(config)
    structure = plan.workload.models["draft"]["structure"]
    assert structure["ffn_type"] == "qwen3_moe"
    assert structure["num_experts"] == 8
    assert structure["num_experts_per_tok"] == 2
    assert config.training.fsdp_strategy == "REPLICATE"
    assert config.training.training_num_gpus_per_node == 2
