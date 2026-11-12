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

"""EAGLE4 stages, reference arithmetic and strict assets; GPU gates live separately."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aspec.config.eagle4 import Eagle4Config, eagle4_policy, eagle4_structure
from aspec.config.train_config import load_config
from aspec.draft_artifact import load_native_draft
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.eagle3 import LazyTarget
from aspec.models.eagle4 import Eagle4TrainingModel
from aspec.models.ops.eagle4_candidate import candidate_retrieval_terms
from aspec.planning import compile_execution_plan
from aspec.training.checkpoint import load_initial_draft_weights
from tools import convert_to_hf

ROOT = Path(__file__).parents[2]


def raw_config(stage="draft", **changes):
    raw = dict(architectures=["LlamaForCausalLMEagle4"], hidden_size=16,
               intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
               num_key_value_heads=1, head_dim=8, vocab_size=32, draft_vocab_size=32,
               target_hidden_size=16, target_num_hidden_layers=6,
               target_layer_ids=[1, 4], num_aux_hidden_states=2, num_target_layers=2,
               max_position_embeddings=128, pad_token_id=0, rms_norm_eps=1e-6,
               rope_theta=10000.0, use_rope=False, fc_norm=True, norm_output=True,
               layer_types=["sliding_attention"], sliding_window=4,
               efficient_lm_head_config={"type": "low_rank", "rank": 4, "num_candidates": 8},
               candidate_teacher_top_k=4, candidate_loss_chunk_size=3,
               eagle4_training_stage=stage)
    raw.update(changes)
    return raw


def model(stage="draft", **changes):
    draft = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw_config(stage, **changes)))
    draft.configure_training_stage()
    return draft


@pytest.mark.parametrize("changes", [
    {"num_hidden_layers": 2}, {"draft_vocab_size": 16}, {"lm_head_rank_divisor": 2},
    {"efficient_lm_head_config": {"type": "svd", "rank": 4, "num_candidates": 8}},
    {"efficient_lm_head_config": {"type": "low_rank", "rank": 0, "num_candidates": 8}},
    {"eagle4_training_stage": "joint"}, {"use_rope": "false"},
    {"enable_2hs": True}, {"candidate_teacher_top_k": 33},
])
def test_invalid_eagle4_requests_fail_closed(changes):
    with pytest.raises(ValueError, match="EAGLE4"):
        Eagle4Config(**raw_config(**changes))


def test_public_recipes_plan_as_separate_stages():
    from aspec.training.eagle4_trainer import Eagle4Trainer
    from aspec.training.trainer_actor import _trainer_class_for_config
    for stage, path in [("draft", "configs/recipes/qwen3_8b_eagle4.yaml"),
                        ("candidate_head", "configs/recipes/qwen3_8b_eagle4_candidate.yaml")]:
        cfg = load_config(str(ROOT / path), save_snapshot=False)
        plan = compile_execution_plan(cfg)
        assert cfg.model.trust_remote_code is False
        assert plan.workload.objective["family"] == "eagle4"
        assert plan.workload.objective["eagle4"]["stage"] == stage
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "model.freeze_lm_head", True)
        frozen = compile_execution_plan(cfg)
        assert "lm_head" in frozen.workload.objective["frozen_parts"]
    assert _trainer_class_for_config(Eagle4Config(**raw_config())) is Eagle4Trainer


@pytest.mark.parametrize("attn_norm,post_norm", [(False, False), (True, True)])
def test_backbone_matches_explicit_upstream_fusion_residual_and_feedback(attn_norm, post_norm):
    torch.manual_seed(4)
    draft = model(attn_norm=attn_norm, enable_post_mlp_norm=post_norm)
    emb, hidden = torch.randn(1, 7, 16), torch.randn(1, 7, 16, requires_grad=True)
    positions = torch.arange(7)[None]
    mask = draft.prepare_decoder_attention_mask(torch.ones(1, 7), hidden, 1, 7, 0)
    layer = draft.midlayer
    fused = torch.cat((layer.input_layernorm(emb), layer.hidden_norm(hidden)), -1)
    residual = hidden
    if attn_norm:
        residual = layer.eh_proj(fused)
        fused = layer.pre_attention_layernorm(residual)
    attended = layer.self_attn(fused, attention_mask=mask, position_ids=positions)[0]
    expected = residual + attended
    expected = expected + layer.mlp(layer.post_attention_layernorm(expected))
    actual = layer(emb, hidden, attention_mask=mask, position_ids=positions)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(draft.next_depth_hidden(actual), draft.norm(actual))
    a = torch.autograd.grad(actual.square().sum(), hidden)[0]
    b = torch.autograd.grad(expected.square().sum(), hidden)[0]
    torch.testing.assert_close(a, b)


def test_candidate_chunked_kl_and_all_gradients_match_direct_formula():
    torch.manual_seed(5)
    hidden = torch.randn(11, 16, requires_grad=True)
    head = torch.randn(32, 16, requires_grad=True)
    down, up = torch.randn(4, 16, requires_grad=True), torch.randn(32, 4, requires_grad=True)
    a = candidate_retrieval_terms(hidden, head, down, up,
                                 teacher_top_k=4, num_candidates=8, chunk_size=3)
    values, ids = F.linear(hidden.detach(), head.detach()).topk(4, -1)
    p = values.softmax(-1)
    q = F.linear(F.linear(hidden.detach(), down), up).log_softmax(-1)
    b = (p * (p.log() - q.gather(-1, ids))).sum()
    torch.testing.assert_close(a[0], b)
    ga = torch.autograd.grad(a[0], (down, up), retain_graph=True)
    gb = torch.autograd.grad(b, (down, up))
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y)
    assert hidden.grad is head.grad is None
    assert a[2] == 11 and 0 <= a[1] <= 11 and 0 <= a[3] <= 11


def test_candidate_freezes_body_for_sixteen_updates_and_exactly_rescores_rows():
    torch.manual_seed(6)
    draft = model("candidate_head")
    draft.initialize_candidate_head()
    body = {k: v.clone() for k, v in draft.state_dict().items()
            if not k.startswith("efficient_lm_head") and k != "candidate_head_steps"}
    opt = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=1e-3)
    hidden = torch.randn(1, 9, 16)
    wrapper = Eagle4TrainingModel(draft)
    for _ in range(16):
        opt.zero_grad()
        loss = wrapper._calculate_loss(hidden, LazyTarget(None, None), torch.ones(1, 9),
                                       0, 9, None, None, None, None)[0] / 9
        loss.backward()
        opt.step()
    for k, value in body.items():
        torch.testing.assert_close(value, draft.state_dict()[k], atol=0, rtol=0)
    assert all(p.grad is None for n, p in draft.named_parameters()
               if not n.startswith("efficient_lm_head"))
    ids, logits = draft.score_candidates(hidden)
    torch.testing.assert_close(logits, F.linear(hidden, draft.lm_head.weight).gather(-1, ids))
    assert all(not child.training for n, child in draft.named_children()
               if not n.startswith("efficient_lm_head"))


@pytest.mark.parametrize("stage", ["draft", "candidate_head"])
def test_native_roundtrip_preserves_stage_and_all_tensors(tmp_path, monkeypatch, stage):
    torch.manual_seed(7)
    draft = model(stage)
    if stage == "candidate_head":
        draft.candidate_head_steps.fill_(16)
    source = tmp_path / "iter_0000016"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").touch()
    (source / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "model_state.model.draft_model." + k: v for k, v in draft.state_dict().items()})
    destination = tmp_path / "asset"
    manifest = convert_to_hf.convert_checkpoint(input_dir=source, output_dir=destination,
                                                artifact_format="aspec")
    assert manifest["eagle4"]["candidate_head_trained"] is (stage == "candidate_head")
    restored = load_native_draft(destination)
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
        assert restored.state_dict()[key].dtype == value.dtype
    next_stage = model("candidate_head")
    load_initial_draft_weights(next_stage, str(destination))
    for key, value in draft.state_dict().items():
        if not key.startswith("efficient_lm_head"):
            torch.testing.assert_close(next_stage.state_dict()[key], value, atol=0, rtol=0)
    data = json.loads((destination / "export_manifest.json").read_text())
    data["eagle4"]["candidate_optimizer_steps"] = 0 if stage == "candidate_head" else 1
    (destination / "export_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_native_draft(destination)


def test_untrained_candidate_export_and_serving_claim_are_rejected(tmp_path, monkeypatch):
    draft = model("candidate_head")
    source = tmp_path / "iter_0000000"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").touch()
    (source / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + k: v for k, v in draft.state_dict().items()})
    with pytest.raises(ValueError, match="candidate"):
        convert_to_hf.convert_checkpoint(input_dir=source, output_dir=tmp_path / "bad",
                                         artifact_format="aspec")
    with pytest.raises(NotImplementedError, match="aspec"):
        convert_to_hf._serving_config(draft.config.to_dict())


def test_router_control_precision_and_failed_or_empty_step():
    from aspec.models.draft.eagle4_moe import Eagle4Router
    cfg = Eagle4Config(**raw_config(enable_moe=True, n_routed_experts=4,
                                    num_experts_per_tok=2, moe_intermediate_size=16))
    router = Eagle4Router(cfg)
    router.e_score_correction_bias.fill_(0.2561234)
    expected = router.e_score_correction_bias.clone()
    router.to(torch.bfloat16)
    torch.testing.assert_close(router.e_score_correction_bias, expected, rtol=0, atol=0)
    router.update_bias()
    torch.testing.assert_close(router.e_score_correction_bias, expected)
    router.load.copy_(torch.tensor([10, 2, 0, 0]))
    router.update_bias(apply_update=False)
    torch.testing.assert_close(router.e_score_correction_bias, expected)
    assert router.load.sum() == 0
    router.load.copy_(torch.tensor([10, 2, 0, 0]))
    router.update_bias()
    torch.testing.assert_close(router.e_score_correction_bias - expected,
                               torch.tensor([-0.001, 0.001, 0.001, 0.001]), atol=2e-8, rtol=0)


def test_moe_contract_matches_explicit_supported_upstream_structures():
    cfg = raw_config(enable_moe=True, n_routed_experts=4, num_experts_per_tok=2,
                     moe_intermediate_size=16, n_group=2, topk_group=1)
    assert eagle4_structure(cfg)["ffn_type"] == "eagle4_moe"
    with pytest.raises(ValueError, match="geometry"):
        eagle4_structure(dict(cfg, num_experts_per_tok=3))
    assert eagle4_policy(cfg)["stage"] == "draft"


@pytest.mark.parametrize("changed", ["stage", "rank", "teacher_top_k"])
def test_exact_resume_keeps_candidate_stage_and_objective(changed):
    from argparse import Namespace

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    original = raw_config("candidate_head")
    args = Namespace(draft_algorithm="eagle4", draft_model_config_obj=Eagle4Config(**original),
                     world_size=1, ttt_length=3, loss_type="forward_kl",
                     lk_eta=3.0, ploss_weights=[1.0, 0.8, 0.64])
    metadata = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    changed_config = dict(original)
    if changed == "stage":
        changed_config["eagle4_training_stage"] = "draft"
    elif changed == "rank":
        changed_config["efficient_lm_head_config"] = {**original["efficient_lm_head_config"], "rank": 2}
    elif changed == "teacher_top_k":
        changed_config["candidate_teacher_top_k"] = 3
    args.draft_model_config_obj = Eagle4Config(**changed_config)
    with pytest.raises(RuntimeError):
        _validate_exact_resume_topology(args, metadata)


def test_meta_rank_configures_freezing_without_svd(monkeypatch):
    from argparse import Namespace

    from aspec.training.eagle4_trainer import Eagle4Trainer

    with torch.device("meta"):
        draft = model("candidate_head")
    trainer = object.__new__(Eagle4Trainer)
    trainer.args = Namespace(initial_draft_model_path="draft-asset")
    monkeypatch.setattr(draft, "initialize_candidate_head", lambda **_: pytest.fail("SVD on meta rank"))
    trainer._prepare_algorithm_training(draft)
    assert all(p.requires_grad == name.startswith("efficient_lm_head")
               for name, p in draft.named_parameters())


def test_router_counts_commit_once_not_during_checkpoint_recomputation(monkeypatch):
    from aspec.models.draft.llama3_eagle import LlamaRMSNorm
    from aspec.utils.tensor import padding

    fn = LlamaRMSNorm.forward
    monkeypatch.setattr(LlamaRMSNorm, "forward", getattr(fn, "_torchdynamo_orig_callable", fn))
    draft = model(enable_moe=True, n_routed_experts=4, num_experts_per_tok=2,
                  moe_intermediate_size=16, n_shared_experts=1)
    wrapper = Eagle4TrainingModel(draft, length=3, gradient_checkpointing=True)
    def local_loss(hidden_states, **kwargs):
        value = hidden_states.square().sum()
        return value, value.detach() * 0, value.new_tensor(8), value.detach() * 0
    monkeypatch.setattr(wrapper, "_calculate_loss", local_loss)
    ids = torch.randint(1, 31, (1, 8))
    result = wrapper(input_ids=padding(ids, left=False), attention_mask=torch.ones_like(ids),
                     target=LazyTarget(None, None), loss_mask=torch.ones_like(ids),
                     hidden_states=torch.randn(1, 8, 32))
    before = draft.midlayer.mlp.gate.load.clone()
    assert before.sum() == 2 * (7 + 6 + 5)
    sum(result[0]).backward()
    torch.testing.assert_close(draft.midlayer.mlp.gate.load, before, atol=0, rtol=0)


def test_inactive_feedback_norm_is_not_owned_by_optimizer():
    draft = model(norm_output=True, enable_post_mlp_norm=True)
    assert not draft.midlayer.post_mlp_layernorm.weight.requires_grad
    assert draft.norm.weight.requires_grad


def test_frozen_candidate_backbone_does_not_request_activation_recomputation():
    from argparse import Namespace

    from aspec.training.eagle4_trainer import Eagle4Trainer

    trainer = object.__new__(Eagle4Trainer)
    trainer.args = Namespace(ttt_length=3, attention_backend="sdpa", gradient_checkpointing=True)
    assert trainer._create_training_model(model()).gradient_checkpointing
    assert not trainer._create_training_model(model("candidate_head")).gradient_checkpointing


def test_native_bf16_moe_retains_exact_fp32_router_bias(tmp_path, monkeypatch):
    draft = model(enable_moe=True, n_routed_experts=4, num_experts_per_tok=2,
                  moe_intermediate_size=16, n_shared_experts=1).bfloat16()
    draft.midlayer.mlp.gate.e_score_correction_bias.fill_(0.2561234)
    source = tmp_path / "iter_0000016"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").touch()
    (source / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + k: v for k, v in draft.state_dict().items()})
    destination = tmp_path / "asset"
    convert_to_hf.convert_checkpoint(input_dir=source, output_dir=destination, artifact_format="aspec")
    loaded = load_native_draft(destination)
    for key, value in draft.state_dict().items():
        assert value.dtype == loaded.state_dict()[key].dtype
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("namespace", ["train", "eval"])
def test_candidate_metrics_do_not_claim_decoder_acceptance(namespace):
    from aspec.controller.eval import eval_selection_metric
    from aspec.training.eagle4_trainer import Eagle4Trainer

    trainer = object.__new__(Eagle4Trainer)
    trainer.draft_model = model("candidate_head")
    result = trainer._metric_semantics({
        f"{namespace}/avg_acc": 0.8, f"{namespace}/avg_alpha": 0.7,
        f"{namespace}/alpha_0": 0.9, f"{namespace}/simulated_acc_len": 2.5,
    }, namespace)
    assert f"{namespace}/simulated_acc_len" not in result
    assert f"{namespace}/avg_alpha" not in result
    assert result[f"{namespace}/candidate_top1_recall"] == 0.8
    assert result[f"{namespace}/candidate_teacher_mass"] == 0.7
    assert result[f"{namespace}/candidate_teacher_mass_0"] == 0.9
    assert eval_selection_metric({"eval/candidate_top1_recall": 0.8}) == "eval/candidate_top1_recall"
    assert eval_selection_metric({"eval/simulated_acc_len": 2.5}) == "eval/simulated_acc_len"

@pytest.mark.parametrize("local_count,global_count,finite", [(0, 4, True), (4, 4, True), (0, 0, True), (0, 4, False)])
def test_candidate_step_proof_is_group_wide(monkeypatch, local_count, global_count, finite):
    from aspec.training.eagle4_trainer import Eagle4Trainer

    trainer = object.__new__(Eagle4Trainer)
    trainer.draft_model = model("candidate_head")
    trainer.dp_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    def all_reduce(value, *, group):
        assert group is trainer.dp_group
        value.fill_(global_count)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    trainer._after_optimizer_step([{"acc_counts": torch.tensor([float(local_count)])}],
                                  torch.tensor(1.0 if finite else float("inf")))
    assert int(trainer.draft_model.candidate_head_steps) == int(global_count > 0 and finite)


def test_meta_router_can_materialize_without_losing_control_dtype():
    from aspec.models.draft.eagle4_moe import Eagle4Router

    cfg = Eagle4Config(**raw_config(enable_moe=True, n_routed_experts=4,
                                    num_experts_per_tok=2, moe_intermediate_size=16))
    with torch.device("meta"):
        router = Eagle4Router(cfg)
    router.to_empty(device="cpu")
    assert router.e_score_correction_bias.device.type == "cpu"
    assert router.e_score_correction_bias.dtype == torch.float32


def test_candidate_resume_can_resize_workspace_but_not_objective():
    from argparse import Namespace

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    original = raw_config("candidate_head")
    args = Namespace(draft_algorithm="eagle4", draft_model_config_obj=Eagle4Config(**original),
                     world_size=1, ttt_length=3, loss_type="forward_kl",
                     lk_eta=3.0, ploss_weights=[1.0, 0.8, 0.64])
    saved = _logical_training_topology(args, 1)
    # Accept checkpoints from the initial implementation which also serialized workspace.
    saved["training_objective"]["eagle4"]["loss_chunk_size"] = original["candidate_loss_chunk_size"]
    args.draft_model_config_obj = Eagle4Config(**{**original, "candidate_loss_chunk_size": 2})
    _validate_exact_resume_topology(args, {"world_size": 1, "training_topology": saved})
