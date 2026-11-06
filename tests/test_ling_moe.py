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

"""Ling routing/FFN/objective/control-state regression, no GPU dependency."""
import copy
import json
import runpy
from pathlib import Path
from unittest import mock

import pytest
import torch

from aspec.config.draft_structure import resolve_draft_structure
from aspec.config.train_config import load_config
from aspec.models.dflash import DFlashModel
from aspec.models.draft.dflash import DFlashAttention, DFlashConfig, DFlashDraftModel
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.draft.ling_moe import LingDraftMoE, ling_router_terms, update_ling_bias
from aspec.models.dspark import DSparkModel
from aspec.planning import compile_execution_plan
from aspec.training.dflash_trainer import DFlashTrainer
from tools.convert_to_hf import _serving_config

REFERENCE = runpy.run_path(str(Path(__file__).with_name("reference_ling_moe.py")))
SETTINGS = dict(ffn_type="ling_moe", num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=12, num_shared_experts=1,
                moe_shared_expert_intermediate_size=10, n_group=2, topk_group=1,
                norm_topk_prob=True, moe_router_enable_expert_bias=True,
                moe_router_bias_update_rate=0.001, bias_zero_mean_update=True,
                router_aux_loss_coef=0, router_z_loss_coef=0.0000035, hidden_act="silu")


def config(cls=DSparkConfig, **changes):
    return cls(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
               num_attention_heads=2, num_key_value_heads=1, vocab_size=31,
               max_position_embeddings=64, num_target_layers=2,
               target_hidden_size=16, target_num_hidden_layers=12,
               target_layer_ids=[1, 9], mask_token_id=30, markov_rank=4,
               **{**SETTINGS, **changes})


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("shared", [1, 2])
def test_official_ling_outputs_input_router_and_expert_gradients(top_k, shared):
    torch.manual_seed(15)
    cfg = config(num_experts_per_tok=top_k, num_shared_experts=shared)
    actual = LingDraftMoE(cfg)
    reference = REFERENCE["BailingMoeV3SparseMoeBlock"](cfg)
    with torch.no_grad():
        actual.gate.expert_bias.copy_(torch.tensor([.17, -.3, .11, .09, -.1, .2, .01, -.03]))
    reference.load_state_dict(actual.state_dict(), strict=True)
    x = torch.randn(2, 3, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    result, (logits, selected) = actual(x)
    expected, (ref_logits, ref_selected) = reference(other)
    torch.testing.assert_close(selected.reshape_as(ref_selected), ref_selected, rtol=0, atol=0)
    torch.testing.assert_close(logits.reshape_as(ref_logits), ref_logits)
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    refs = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, refs[name].grad, atol=3e-6, rtol=2e-5)
    assert actual.gate.expert_bias.grad is None


def test_expert_bias_selects_only_and_top1_is_not_renormalized():
    model = LingDraftMoE(config(num_experts_per_tok=1))
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.expert_bias[7] = 10
    selected, weights, _ = model.gate(torch.ones(2, 16))
    assert selected.tolist() == [[7], [7]]
    torch.testing.assert_close(weights, torch.full((2, 1), 0.5 * 2.5))


def test_bf16_cast_keeps_exact_fp32_bias_and_checkpoint_roundtrip():
    model = DSparkDraftModel(config())
    for layer in model.layers:
        layer.mlp.gate.expert_bias.copy_(torch.arange(8) * 0.00123)
    expected = copy.deepcopy(model.state_dict())
    model.to(torch.bfloat16)
    restored = DSparkDraftModel(DSparkConfig.from_dict(model.config.to_dict())).to(torch.bfloat16)
    restored.load_state_dict(model.state_dict(), strict=True)
    for i, layer in enumerate(restored.layers):
        bias = layer.mlp.gate.expert_bias
        assert bias.dtype == torch.float32
        torch.testing.assert_close(bias, expected[f"layers.{i}.mlp.gate.expert_bias"], atol=0, rtol=0)


def test_per_layer_swiglu_clamp_matches_official_sglang_arithmetic():
    cfg = config(expert_swiglu_limit_list=[0, .03], share_expert_swiglu_limit_list=[0, .07])
    model = LingDraftMoE(cfg, layer_idx=1)
    x = torch.randn(3, 16, requires_grad=True) * 50
    for expert in (model.experts[0], model.shared_experts):
        gate = torch.nn.functional.silu(expert.gate_proj(x)).clamp(max=expert.limit)
        up = expert.up_proj(x).clamp(min=-expert.limit, max=expert.limit)
        expected = expert.down_proj(gate * up)
        torch.testing.assert_close(expert(x), expected)
    # The selected clamp belongs to Draft layer index, not a Target capture ID.
    draft = DSparkDraftModel(cfg)
    assert draft.layers[0].mlp.experts[0].limit == 0
    assert draft.layers[1].mlp.shared_experts.limit == .07


@pytest.mark.parametrize("all_masked", [False, True])
def test_z_loss_counts_masking_and_no_qwen_auxiliary(all_masked):
    torch.manual_seed(5)
    logits = torch.randn(6, 8, requires_grad=True)
    selected = torch.tensor([[0, 1], [0, 1], [2, 3], [4, 5], [6, 7], [0, 7]])
    mask = torch.tensor([1, 1, 0, 1, 0, 0]).float()
    if all_masked:
        mask.zero_()
    numerator, metrics = ling_router_terms(((logits, selected),), mask, mask.sum(), .03)
    reference = ((logits.logsumexp(-1).square() * mask).sum() / mask.sum().clamp_min(1)) * mask.sum() * .03
    torch.testing.assert_close(numerator, reference)
    assert "router_aux_loss" not in metrics
    torch.testing.assert_close(metrics["_ling_router_counts"].sum(), mask.sum() * 2)
    numerator.backward()
    assert torch.count_nonzero(logits.grad[mask == 0]) == 0
    if all_masked:
        assert numerator == 0


@pytest.mark.parametrize("zero_mean", [False, True])
@pytest.mark.parametrize("apply_update", [False, True])
def test_bias_update_matches_ling_megatron_rule(zero_mean, apply_update):
    layers = [LingDraftMoE(config()), LingDraftMoE(config())]
    counts = torch.tensor([[8., 5, 4, 3, 0, 0, 0, 0], [0.] * 8])
    reference = (counts.mean(-1, keepdim=True) - counts).sign()
    if zero_mean:
        reference -= reference.mean(-1, keepdim=True)
    update_ling_bias(layers, counts, .001, zero_mean, apply_update=apply_update)
    torch.testing.assert_close(layers[0].gate.expert_bias, reference[0] * .001 * apply_update)
    assert layers[1].gate.expert_bias.count_nonzero() == 0


def test_trainer_accumulates_counts_once_per_step_and_no_eval_mutation():
    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = DSparkDraftModel(config())
    trainer.dp_group = None
    counts = torch.tensor([[8., 0, 0, 0, 0, 0, 0, 0]] * 2)
    metrics = [{"_ling_router_counts": counts}, {"_ling_router_counts": counts.roll(1, -1)}]
    trainer._update_ling_router_bias(metrics, torch.tensor(1.))
    expected = (counts + counts.roll(1, -1)).mean(-1, keepdim=True) - counts - counts.roll(1, -1)
    expected = expected.sign()
    expected -= expected.mean(-1, keepdim=True)
    bias = torch.stack([layer.mlp.gate.expert_bias for layer in trainer.draft_model.layers])
    torch.testing.assert_close(bias, expected * .001)
    trainer._update_ling_router_bias(metrics, torch.tensor(float("inf")))
    torch.testing.assert_close(torch.stack([layer.mlp.gate.expert_bias for layer in trainer.draft_model.layers]), bias)
    model = trainer.draft_model.layers[0].mlp
    model.eval()
    before = model.gate.expert_bias.clone()
    model(torch.randn(1, 3, 16))
    torch.testing.assert_close(model.gate.expert_bias, before, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="missing router"):
        trainer._update_ling_router_bias([{}], torch.tensor(1.))


@pytest.mark.parametrize("algorithm", ["dspark", "dflash"])
def test_real_wrappers_train_ling_router_shared_expert_and_preserve_semantics(algorithm):
    cls, wrapper = (DSparkDraftModel, DSparkModel) if algorithm == "dspark" else (DFlashDraftModel, DFlashModel)
    draft = cls(config(DSparkConfig if algorithm == "dspark" else DFlashConfig))
    draft.freeze_embedding()
    model = wrapper(draft, block_size=3, num_anchors=2)
    def cheap_attention(self, draft_hidden, context_hidden, **kwargs):
        return self.o_proj(self.q_proj(draft_hidden)) + context_hidden.mean(1, keepdim=True)
    batch = dict(input_ids=torch.randint(0, 30, (1, 12)), hidden_states_list=torch.randn(1, 12, 32),
                 loss_mask=torch.ones(1, 12), lm_head_weight=torch.randn(31, 16),
                 last_hidden_states=torch.randn(1, 12, 16), injected_anchors=torch.tensor([[2, 5]]),
                 injected_keep_mask=torch.ones(1, 2, dtype=torch.bool))
    with mock.patch.object(DFlashAttention, "forward", cheap_attention):
        result = model(**batch)
    assert result[5]["_ling_router_counts"].shape == (2, 8)
    assert "router_aux_loss" not in result[5] and "router_z_loss" in result[5]
    result[0].backward()
    assert draft.embed_tokens.weight.grad is None
    for layer in draft.layers:
        assert all(p.grad is not None for p in layer.mlp.parameters())
        assert layer.mlp.gate.weight.grad.abs().sum() > 0
        assert layer.mlp.shared_experts.down_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("change", [
    {"n_group": 3}, {"n_group": 8}, {"topk_group": 3}, {"num_experts_per_tok": 5},
    {"norm_topk_prob": False}, {"router_aux_loss_coef": .01}, {"score_function": "softmax"},
    {"moe_router_bias_update_rate": -1}, {"bias_zero_mean_update": "yes"},
    {"moe_router_enable_expert_bias": False, "moe_router_bias_update_rate": .001},
    {"router_z_loss_coef": float("nan")}, {"first_k_dense_replace": 1},
    {"expert_swiglu_limit_list": [4]}, {"share_expert_swiglu_limit_list": [-1, 0]},
    {"num_shared_experts": -1}, {"shared_expert_intermediate_size": 7},
    {"scale_router_input": True}, {"router_dtype": "bf16"},
])
def test_ling_invalid_combinations_fail_before_allocation(change):
    with pytest.raises(ValueError):
        config(**change)


def test_recipe_plan_has_ling_semantics_and_offline_export_contract():
    root = Path(__file__).parents[1]
    recipe = load_config(str(root / "configs/examples/ling3_flash_multi_structure_dspark.yaml"))
    recipe.experiments = []
    assert Path(recipe.dataset.train_data_path).is_file()
    recipe.model.draft_model_config = str(root / "aspec/config/dspark_ling3_flash_moe_draft_config.json")
    plan = compile_execution_plan(recipe)
    structure = plan.workload.models["draft"]["structure"]
    assert structure["ffn_type"] == "ling_moe"
    assert structure["router_aux_loss_coef"] == 0
    assert structure["moe_router_bias_update_rate"] == .001
    raw = json.loads(Path(recipe.model.draft_model_config).read_text())
    assert resolve_draft_structure(raw) == structure
    assert _serving_config(raw)["ffn_type"] == "ling_moe"


def test_ling_moe_mha_swa_override_preserves_structure_and_execution():
    root = Path(__file__).parents[1]
    base_path = root / "aspec/config/dspark_ling3_flash_moe_draft_config.json"
    combo_path = root / "aspec/config/dspark_ling3_flash_moe_swa_draft_config.json"
    base = json.loads(base_path.read_text())
    combo = json.loads(combo_path.read_text())
    assert {key: combo[key] for key in base} == base
    assert combo["sliding_window"] == 128

    recipe = load_config(
        str(root / "configs/examples/ling3_flash_multi_structure_dspark.yaml"), save_snapshot=False
    )
    recipe.experiments = []
    recipe.model.draft_model_config = str(combo_path)
    plan = compile_execution_plan(recipe)
    draft = plan.workload.models["draft"]
    assert draft["structure"]["ffn_type"] == "ling_moe"
    assert draft["attention"]["head_layout"]["type"] == "mha"
    assert draft["attention"]["context_windows"] == [128] * 5
    args = plan.to_runtime_args()
    assert args.global_batch_size == 4
    assert args.num_train_steps == 16 and args.lr_total_steps == 32
    assert args.draft_context_parallel_size == 1 and not args.dflash_packing
    assert args.fsdp_strategy == "REPLICATE"


def test_real_training_entry_updates_bias_after_optimizer_once_and_before_metrics():
    from types import SimpleNamespace

    from aspec.training.trainer import Trainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = DSparkDraftModel(config())
    trainer.model = trainer.draft_model
    trainer.dp_group = None
    trainer.global_step = 0
    trainer._enable_perf_metrics = False
    trainer._work_lifecycle_enabled = False
    trainer.prof = SimpleNamespace(iterate_train_actor=lambda batches: batches)
    trainer._iter_batches_from_queue = lambda n: iter([{}, {}])
    counts = torch.tensor([[8., 0, 0, 0, 0, 0, 0, 0]] * 2)
    trainer._train_step = lambda **kwargs: {"_ling_router_counts": counts}
    trainer._maybe_dump = mock.Mock()
    order = []
    def optimizer_step():
        assert all(layer.mlp.gate.expert_bias.count_nonzero() == 0 for layer in trainer.draft_model.layers)
        order.append("optimizer")
        return torch.tensor(1.)
    trainer.optimizer = SimpleNamespace(step=optimizer_step)
    def metrics(*args, **kwargs):
        assert any(layer.mlp.gate.expert_bias.count_nonzero() > 0 for layer in trainer.draft_model.layers)
        order.append("metrics")
        return {}
    trainer._aggregate_metrics = metrics
    with mock.patch("torch.cuda.Event"), mock.patch("torch.cuda.synchronize"):
        Trainer._train_core_from_queue(trainer, step=0, num_batches=2)
    assert trainer.global_step == 1
    assert order == ["optimizer", "metrics"]
    expected = (counts.mean(-1, keepdim=True) - counts).sign()
    expected -= expected.mean(-1, keepdim=True)
    torch.testing.assert_close(trainer.draft_model.layers[0].mlp.gate.expert_bias, expected[0] * .001)


@pytest.mark.parametrize("ffn_type", ["dense", "qwen3_moe"])
@pytest.mark.parametrize("extra", [
    {"router_z_loss_coef": .001}, {"moe_router_bias_update_rate": .001},
    {"bias_zero_mean_update": True}, {"expert_swiglu_limit_list": [0, 4]},
])
def test_ling_training_settings_are_not_silently_accepted_by_other_ffns(ffn_type, extra):
    raw = {"ffn_type": "dense"} if ffn_type == "dense" else {
        "ffn_type": "qwen3_moe", "num_experts": 8,
        "num_experts_per_tok": 2, "moe_intermediate_size": 12,
    }
    with pytest.raises(ValueError, match="ffn_type=ling_moe"):
        resolve_draft_structure({**raw, **extra})


def test_dense_cannot_silently_enable_expert_bias():
    with pytest.raises(ValueError, match="explicit ffn_type"):
        DFlashConfig(moe_router_enable_expert_bias=True)
