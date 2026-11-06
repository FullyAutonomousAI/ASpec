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

"""Independent DSA score/teacher references and public lifecycle contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from aspec.config.draft_attention import resolve_draft_attention
from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.dsa import (
    DSAForwardMetadata,
    DSALayer,
    DSALightningIndexer,
    _apply_rotary,
    _owner_geometry,
    _rotary_factors,
    build_dsa_token_layout,
)
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from tools.convert_to_hf import _serving_config


def config(stage="sparse", **changes):
    values = dict(
        hidden_size=16, intermediate_size=32, head_dim=4, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        max_position_embeddings=64, mask_token_id=31,
        layer_types=["deepseek_sparse_attention"], dsa_training_stage=stage,
        dsa_token_budget=3, dsa_index_n_heads=2, dsa_index_head_dim=4,
        dsa_q_lora_rank=8, dsa_rope_dim=2, dsa_workspace_bytes=1 << 20,
        markov_rank=4, enable_confidence_head=True,
    )
    return DSparkConfig(**(values | changes))


def inputs():
    torch.manual_seed(1023)
    anchors = torch.tensor([[3, 6]])
    metadata = DSAForwardMetadata(anchors, torch.ones_like(anchors).bool(), 2)
    positions = torch.arange(8)[None]
    draft_positions = torch.tensor([[3, 4, 6, 7]])
    layout = build_dsa_token_layout(metadata, positions, draft_positions,
                                   rope_dim=2, rope_theta=10000.0)
    values = dict(
        draft_hidden=torch.randn(1, 4, 16, requires_grad=True),
        context_hidden=torch.randn(1, 8, 16, requires_grad=True),
        draft_position_ids=draft_positions, metadata=metadata, layout=layout,
        main_query=torch.randn(1, 4, 2, 4, requires_grad=True),
        main_key=torch.randn(1, 12, 1, 4, requires_grad=True), scale=0.5,
    )
    return values


def test_policy_roundtrip_and_uncompressed_identity():
    current = config()
    policy = resolve_draft_attention(current.to_dict())
    assert policy["csa"] is policy["qsa"] is None
    assert policy["dsa"]["draft_block_index_reuse"] is False
    assert policy["dsa"]["indexer_input_detached"] is True
    assert policy["dsa"]["token_budget"] == 3  # no compression divisibility constraint
    assert resolve_draft_attention(DFlashConfig.from_dict(current.to_dict()).to_dict()) == policy


@pytest.mark.parametrize("changes", [
    {"dsa_token_budget": 0}, {"dsa_token_budget": True}, {"dsa_q_lora_rank": 0},
    {"dsa_rope_dim": 3}, {"dsa_rope_dim": 6}, {"dsa_workspace_bytes": -1},
    {"dsa_rope_theta": float("nan")}, {"dsa_indexer_loss_coef": float("inf")},
    {"dsa_training_stage": "unknown"},
    {"dsa_training_stage": "indexer_warmup", "dsa_indexer_loss_coef": 0.0},
    {"layer_types": ["full_attention"]},
    {"num_hidden_layers": 2, "layer_types": ["qwen_sparse_attention", "deepseek_sparse_attention"]},
])
def test_invalid_dsa_configuration_fails_closed(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_rotary_matches_deepseek_indexer_noninterleaved_layout():
    torch.manual_seed(55)
    tensor = torch.randn(1, 3, 2, 6)
    positions = torch.tensor([[0, 2, 7]])
    actual = _apply_rotary(tensor, _rotary_factors(positions, 4, 10000.0))
    pairs = torch.view_as_complex(tensor[..., :4].reshape(1, 3, 2, 2, 2).transpose(-1, -2).contiguous())
    frequency = positions.float()[..., None] * torch.tensor([1.0, 0.01])
    expected = pairs * torch.polar(torch.ones_like(frequency), frequency)[:, :, None]
    torch.testing.assert_close(actual[..., :4], torch.cat((expected.real, expected.imag), dim=-1))
    torch.testing.assert_close(actual[..., 4:], tensor[..., 4:])


def test_fsdp_input_cast_keeps_shared_rotary_cache_in_fp32():
    from torch.distributed.utils import _apply_to_tensors

    layout = inputs()["layout"]
    casted = _apply_to_tensors(lambda tensor: tensor.bfloat16() if tensor.is_floating_point() else tensor, layout)
    assert casted.rotary_factors is layout.rotary_factors
    assert casted.key_rotary[0].dtype == torch.float32


def test_owned_query_adapter_and_layernorm_match_independent_projection_math():
    values = inputs()
    indexer = DSALightningIndexer(config())
    query, key, weights = indexer.project(values["draft_hidden"], values["context_hidden"], values["layout"])
    draft = values["draft_hidden"].detach()
    latent = F.linear(draft, indexer.q_a_proj.weight)
    latent = F.rms_norm(latent, (8,), indexer.q_a_norm.weight, indexer.q_a_norm.eps)
    expected_query = F.linear(latent, indexer.q_b_proj.weight).unflatten(-1, (2, 4))
    expected_key = F.layer_norm(F.linear(values["context_hidden"].detach(), indexer.k_proj.weight),
                                (4,), indexer.k_norm.weight, indexer.k_norm.bias, indexer.k_norm.eps)
    torch.testing.assert_close(query, _apply_rotary(expected_query, values["layout"].query_rotary))
    torch.testing.assert_close(key, _apply_rotary(expected_key.unsqueeze(2), values["layout"].key_rotary).squeeze(2))
    torch.testing.assert_close(weights, F.linear(draft.float(), indexer.weights_proj.weight.float()) * indexer.scale)


def teacher_reference(query, key, metadata, selected):
    target = torch.zeros(1, 4, 8 if selected is None else selected.shape[-1])
    for row in range(4):
        anchor = int(metadata.query_anchors[0, row])
        context = list(range(anchor)) if selected is None else [
            int(index) for index in selected[0, row] if index >= 0
        ]
        own_block = [8 + row // 2 * 2 + offset for offset in range(2)]
        for head in range(2):
            scores = query[0, row, head] @ key[0, context + own_block, 0].T * 0.5
            probability = scores.softmax(-1)[:len(context)]
            if selected is None:
                target[0, row, context] += probability
            else:
                target[0, row, :len(context)] += probability
    return target / target.sum(-1, keepdim=True).clamp_min(1e-12)


@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
def test_weighted_relu_selection_kl_and_detached_ownership(stage):
    values = inputs()
    layer = DSALayer(config(stage), 0)
    query, key, weights = layer.indexer.project(
        values["draft_hidden"], values["context_hidden"], values["layout"]
    )
    scores = torch.zeros(1, 4, 8)
    for head in range(2):
        scores += (query[:, :, head] @ key.transpose(1, 2)).relu() * weights[:, :, head, None]
    torch.testing.assert_close(layer.indexer.scores(query, key, weights), scores)
    selected, loss, observed = layer(**values)
    assert observed["draft_block_index_reuse"] is False
    if stage == "sparse":
        assert selected.shape == (1, 4, 3)
        actual_scores = scores.gather(-1, selected.clamp_min(0))
    else:
        assert selected is None
        actual_scores = scores
    target = teacher_reference(values["main_query"].detach(), values["main_key"].detach(),
                               values["metadata"], selected)
    if selected is None:
        visible = torch.arange(8)[None, None] < values["metadata"].query_anchors[..., None]
    else:
        visible = selected >= 0
    log_prob = actual_scores.masked_fill(~visible, torch.finfo(scores.dtype).min).log_softmax(-1)
    expected = F.kl_div(log_prob, target, reduction="none").sum(-1)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-5)
    loss.sum().backward()
    for name in ("draft_hidden", "context_hidden", "main_query", "main_key"):
        assert values[name].grad is None, name
    assert all(parameter.grad is not None for parameter in layer.indexer.parameters())


def test_token_selection_is_per_query_not_per_block():
    torch.manual_seed(22)
    indexer = DSALightningIndexer(config())
    query = torch.tensor([[[[1.0, 0, 0, 0]], [[0.0, 1, 0, 0]]]])
    key = torch.tensor([[[3.0, 0, 0, 0], [0.0, 3, 0, 0]]])
    scores = indexer.scores(query, key, torch.ones(1, 2, 1))
    assert scores.argmax(-1).tolist() == [[0, 1]]


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
def test_complete_dspark_phase_freeze_includes_markov_and_confidence(stage):
    model = DSparkDraftModel(config(stage))
    for name, parameter in model.named_parameters():
        indexer = ".dsa_layer.indexer." in name
        expected = indexer if stage == "indexer_warmup" else not indexer if stage == "dense" else True
        assert parameter.requires_grad == expected, name


def test_sparse_zero_auxiliary_coefficient_freezes_unused_indexer():
    model = DSparkDraftModel(config(dsa_indexer_loss_coef=0.0))
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (".dsa_layer.indexer." not in name)


def test_dflash_indexer_warmup_needs_no_lm_head_or_main_objective(monkeypatch):
    from aspec.models.dflash import DFlashModel
    from aspec.models.draft.dflash import DFlashDraftModel

    draft = DFlashDraftModel(DFlashConfig.from_dict(config("indexer_warmup").to_dict()))
    wrapper = DFlashModel(draft, block_size=2, num_anchors=2, loss_decay_gamma=0.0, l1_loss_alpha=1.0)
    objective = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    def backbone(*_args, **kwargs):
        kwargs["dsa_losses_out"].append(objective)
        return torch.zeros(1, 4, 16), torch.tensor([[1, 5]]), torch.ones(1, 2).bool(), 2

    monkeypatch.setattr(wrapper, "_draft_backbone", backbone)
    monkeypatch.setattr(wrapper, "_compute_logits", lambda *_a: pytest.fail("Warmup read the LM head"))
    result = wrapper(torch.ones(1, 8, dtype=torch.long), [], torch.ones(1, 8), None)
    torch.testing.assert_close(result[0], torch.tensor(3.0))  # Only non-anchor slots 2 and 4.
    assert not wrapper.uses_target_hidden_states
    assert set(result[5]) == {"dsa_indexer_loss"}
    result[0].backward()
    torch.testing.assert_close(objective.grad, torch.tensor([[0.0, 0.5, 0.0, 0.5]]))


def test_packed_token_layout_never_crosses_documents():
    from aspec.models.draft.qsa import qsa_eligible_groups

    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 1, -1]])
    metadata = DSAForwardMetadata(torch.tensor([[2, 6]]), torch.ones(1, 2).bool(), 2,
                                  docs, torch.tensor([[0, 1]]))
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0]])
    layout = build_dsa_token_layout(metadata, positions, torch.tensor([[2, 3, 3, 4]]),
                                   rope_dim=2, rope_theta=10000.0)
    assert layout.token_indices.shape == (1, 8, 1)
    visible = qsa_eligible_groups(metadata, layout)
    assert visible[0, 0].nonzero().flatten().tolist() == [0, 1]
    assert visible[0, 2].nonzero().flatten().tolist() == [3, 4, 5]


@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
def test_workspace_chunks_preserve_gradients_without_saved_full_context_scores(stage):
    values = inputs()
    large = DSALayer(config(stage), 0)
    small = copy.deepcopy(large)
    small.indexer.workspace_bytes = 1
    _, expected, _ = large(**values)
    expected.sum().backward()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor.shape) or tensor,
                                                lambda tensor: tensor):
        _, actual, _ = small(**values)
        actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    for left, right in zip(large.parameters(), small.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
    assert torch.Size([1, 1, 8]) not in saved  # no saved query-chunk x full-context score/mask


def test_distributed_teacher_batches_collectives_across_kv_heads(monkeypatch):
    """GQA head count must not multiply CP synchronization points."""

    from aspec.models.draft import qsa

    metadata = DSAForwardMetadata(
        torch.tensor([[3, 7]]),
        torch.ones(1, 2, dtype=torch.bool),
        2,
        context_physical_position_ids=torch.arange(4)[None],
    )
    query_indices = torch.tensor([0, 1])
    layout = build_dsa_token_layout(
        metadata,
        torch.arange(4)[None],
        torch.tensor([[3, 4]]),
        rope_dim=2,
        rope_theta=10000.0,
    )
    direct = SimpleNamespace(
        context_position_ids=torch.arange(4)[None],
        local_block_ids=torch.tensor([0]),
        context_length=4,
        num_blocks=2,
        cp_size=2,
        group=object(),
    )
    torch.manual_seed(1031)
    main_query = torch.randn(1, 2, 4, 4)
    main_key = torch.randn(1, 6, 2, 4)
    calls = []

    def identity_all_reduce(value, op=None, group=None):
        calls.append((tuple(value.shape), op, group))

    monkeypatch.setattr(qsa.dist, "all_reduce", identity_all_reduce)
    full, full_valid = qsa._distributed_full_teacher_target(
        main_query=main_query,
        main_key=main_key,
        metadata=metadata,
        layout=layout,
        query_indices=query_indices,
        direct_cp_metadata=direct,
        scale=0.5,
    )
    reference_full = qsa._full_teacher_target(
        main_query=main_query,
        main_key=main_key,
        metadata=metadata,
        layout=layout,
        start=0,
        end=2,
        scale=0.5,
        query_indices=query_indices,
    )
    torch.testing.assert_close(full, reference_full)
    assert full_valid.all()
    assert len(calls) == 3  # max, denominator, normalized group row sum

    calls.clear()
    selected = torch.tensor([[[0, 1], [1, 2]]])
    sparse = qsa._distributed_selected_teacher_target(
        main_query=main_query,
        main_key=main_key,
        metadata=metadata,
        layout=layout,
        local_selected=selected,
        direct_cp_metadata=direct,
        scale=0.5,
        workspace_bytes=1 << 20,
        block_indices=torch.tensor([0, 0]),
    )
    reference_sparse = qsa._selected_teacher_target(
        main_query=main_query,
        main_key=main_key,
        metadata=metadata,
        layout=layout,
        selected=selected,
        start=0,
        end=2,
        scale=0.5,
        query_indices=query_indices,
    )
    torch.testing.assert_close(sparse, reference_sparse)
    assert len(calls) == 3  # max, denominator, selected-target sum


def test_serving_export_does_not_silently_erase_dsa():
    with pytest.raises(NotImplementedError, match="DSA"):
        _serving_config(config().to_dict())


def test_dsa_example_configs_keep_same_identity_across_phase_boundaries():
    root = Path(__file__).parents[1]
    configs = []
    for stage, name in (("dense", "dense"), ("indexer_warmup", "indexer"), ("sparse", "sparse")):
        path = root / f"aspec/config/dspark_qwen3_8b_dsa_{name}_draft_config.json"
        raw = json.loads(path.read_text())
        assert raw["dsa_training_stage"] == stage
        raw.pop("dsa_training_stage")
        configs.append(raw)
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_plan_preserves_dsa_and_packing_without_compression_alignment(monkeypatch, stage, cp):
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    recipe = root / "configs/examples/qwen3_8b_dsa_dspark.yaml"
    run = load_config(str(recipe), save_snapshot=False)
    phase = "indexer" if stage == "indexer_warmup" else stage
    run.model.draft_model_config = str(root / f"aspec/config/dspark_qwen3_8b_dsa_{phase}_draft_config.json")
    run.training.draft_context_parallel_size = cp
    run.training.training_num_gpus_per_node = cp
    if stage == "indexer_warmup":
        run.inference.store_last_hidden_states = False
    run.mooncake.source_gpu_ring_slots = cp * 2
    plan = compile_execution_plan(run, config_path=str(recipe))
    assert plan.workload.models["draft"]["attention"]["dsa"]["training_stage"] == stage
    assert plan.to_runtime_args().draft_context_shard_alignment == 1
    decision = next(item for item in plan.capabilities if item.name == "draft_attention_execution")
    assert decision.requirement == "required"
    thd = next(item for item in plan.capabilities if item.name == "draft_thd_execution").resolved
    assert thd["fa4_mode"] == ("varlen_selected" if stage == "sparse" else "fixed")
    assert thd["backward_strategy"] == (
        "not_required_frozen_backbone" if stage == "indexer_warmup" else
        "fa4_varlen_selected" if stage == "sparse" else "fa4_fixed_sparse"
    )


def test_mixed_dsa_thd_contract_records_both_kernel_modes():
    from aspec.config.draft_attention import draft_thd_contract

    policy = resolve_draft_attention(config(num_hidden_layers=2, layer_types=[
        "full_attention", "deepseek_sparse_attention",
    ]).to_dict())
    actual = draft_thd_contract(policy)
    assert actual["fa4_mode"] == actual["backward_strategy"] == "mixed"
    assert actual["fa4_modes"] == ["fixed", "varlen_selected"]
    assert actual["backward_strategies"] == ["fa4_fixed_sparse", "fa4_varlen_selected"]


def test_dense_dsa_direct_attention_retains_layer_policy_without_thd_metadata():
    from aspec.models.ops.direct_cp import DirectCPMetadata

    values = inputs()
    draft = DSparkDraftModel(config("dense"))
    metadata = values["metadata"]
    positions = torch.arange(8)[None]
    direct = DirectCPMetadata(
        anchor_positions=metadata.anchor_positions,
        block_keep_mask=metadata.block_keep_mask,
        context_position_ids=positions,
        block_size=2, cp_rank=0, cp_size=1, group=None,
        sequence_length=8, backend="math",
    )
    # Non-packed evaluation need not emit a THD kernel observation. The real
    # sparse-family layer still produces its executed stage/policy descriptor.
    draft(
        draft_input_ids=None,
        context_feature=values["context_hidden"],
        noise_embedding=values["draft_hidden"],
        draft_position_ids=values["draft_position_ids"],
        context_position_ids=positions,
        direct_cp_metadata=direct,
        dsa_metadata=metadata,
    )
    observed = draft.layers[0].self_attn.attention_observation
    assert observed["type"] == "deepseek_sparse_attention"
    assert observed["training_stage"] == "dense"
    assert "qkv_format" not in observed
    assert draft.last_attention_observations == (observed,)


@pytest.mark.parametrize("kind", ["csa", "qsa", "dsa"])
@pytest.mark.parametrize("stage,mode", [
    ("dense", "fixed"),
    ("indexer_warmup", "fixed"),
    ("sparse", "varlen_selected"),
])
def test_sparse_family_thd_contract_matches_actual_stage(kind, stage, mode):
    from aspec.config.draft_attention import draft_thd_contract

    fields = {
        "csa": dict(layer_types=["compressed_sparse_attention"], sliding_window=8,
                    compress_ratios=[4], csa_training_stage=stage, index_topk=2,
                    index_n_heads=2, index_head_dim=4),
        "qsa": dict(layer_types=["qwen_sparse_attention"], qsa_training_stage=stage,
                    qsa_token_budget=4, qsa_compress_ratio=2,
                    qsa_index_n_heads=2, qsa_index_head_dim=4, qsa_rope_dim=2),
        "dsa": dict(layer_types=["deepseek_sparse_attention"], dsa_training_stage=stage,
                    dsa_token_budget=4, dsa_index_n_heads=2, dsa_index_head_dim=4,
                    dsa_q_lora_rank=8, dsa_rope_dim=2),
    }[kind]
    policy = resolve_draft_attention({"num_hidden_layers": 1, **fields})
    actual = draft_thd_contract(policy)
    assert actual["fa4_mode"] == mode
    assert actual["backward_strategy"] == (
        "not_required_frozen_backbone"
        if stage == "indexer_warmup"
        else "fa4_varlen_selected" if stage == "sparse" else "fa4_fixed_sparse"
    )


@pytest.mark.parametrize("layout,maximum", [("contiguous", 6), ("zigzag", 6)])
def test_uneven_cp_search_geometry_is_rank_independent_and_cached(layout, maximum):
    from aspec.models.ops.direct_cp import DirectCPMetadata, build_context_positions

    for rank in range(4):
        direct = DirectCPMetadata(
            anchor_positions=torch.tensor([[3, 7, 11, 15, 17, 18, 19, 20]]),
            block_keep_mask=torch.ones(1, 8).bool(),
            context_position_ids=build_context_positions(22, rank, 4, layout=layout)[None],
            block_size=2, cp_rank=rank, cp_size=4, group=object(), sequence_length=22, layout=layout,
        )
        key = torch.empty(1, direct.context_length, 4)
        actual = _owner_geometry(key, direct)
        assert actual == (22, maximum)
        assert _owner_geometry(key, direct) is actual


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
def test_trainer_reports_actual_thd_mode_and_frozen_warmup(stage):
    from aspec.config.draft_attention import draft_thd_contract
    from aspec.training.dflash_trainer import DFlashTrainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = DSparkDraftModel(config(stage))
    policy = resolve_draft_attention(trainer.draft_model.config.to_dict())
    contract = draft_thd_contract(policy)
    trainer.args = SimpleNamespace(attention_backend="fa4", draft_context_parallel_size=1)
    trainer.data_fetcher = SimpleNamespace(get_feature_contract_status=lambda: {})
    trainer.dflash = SimpleNamespace(attention_policy_observed=policy)
    trainer.draft_model.layers[0].self_attn.attention_observation = {
        "qkv_format": "thd", "fa4_mode": contract["fa4_mode"],
        "backward_strategy": "fa4_varlen_selected" if stage == "sparse" else "fa4_fixed_sparse",
        "backward_observed": stage != "indexer_warmup", "empty_local_partition": False,
    }
    execution = trainer.get_feature_contract_status()["execution_metadata"]
    assert execution["draft_thd_fa4_mode"] == [contract["fa4_mode"]]
    assert execution["draft_attention_backward"] == [contract["backward_strategy"]]
    if stage == "indexer_warmup":
        assert execution["draft_body_frozen"] is True
        assert execution["draft_attention_backward_required"] is False
        assert execution["draft_thd_backward_validated"] == [False]


def test_exact_resume_preserves_dsa_stage_but_not_workspace():
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    original = SimpleNamespace(world_size=1, draft_model_config_obj=config("dense"))
    metadata = {"world_size": 1, "training_topology": _logical_training_topology(original, 1)}
    _validate_exact_resume_topology(original, metadata)
    workspace = SimpleNamespace(world_size=1, draft_model_config_obj=config("dense", dsa_workspace_bytes=256))
    _validate_exact_resume_topology(workspace, metadata)
    different_stage = SimpleNamespace(world_size=1, draft_model_config_obj=config("sparse"))
    with pytest.raises(RuntimeError, match="topology"):
        _validate_exact_resume_topology(different_stage, metadata)
    with pytest.raises(RuntimeError, match="DSA policy"):
        _validate_exact_resume_topology(original, {"world_size": 1})


def test_token_layout_and_rotary_are_reused_only_within_one_model_forward(monkeypatch):
    import aspec.models.draft.dsa as module

    model = DSparkDraftModel(config(num_hidden_layers=2, layer_types=["deepseek_sparse_attention"] * 2))
    values = inputs()
    layouts, calls = [], []
    original = module.build_dsa_token_layout

    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_dsa_token_layout", observe)
    for layer in model.layers:
        monkeypatch.setattr(layer, "forward", lambda **kwargs: (
            layouts.append(kwargs["dsa_layout"]) or kwargs["draft_hidden"]
        ))
    for _ in range(2):
        model(draft_input_ids=None, context_feature=values["context_hidden"],
              draft_position_ids=values["draft_position_ids"],
              context_position_ids=torch.arange(8)[None], noise_embedding=values["draft_hidden"],
              dsa_metadata=values["metadata"])
    assert len(calls) == 2
    assert layouts[0] is layouts[1] and layouts[2] is layouts[3]
    assert layouts[0] is not layouts[2]


def test_frozen_ling_main_model_does_not_update_router_bias_during_dsa_warmup():
    from aspec.training.dflash_trainer import DFlashTrainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = SimpleNamespace(config=SimpleNamespace(
        ffn_type="ling_moe", dsa_training_stage="indexer_warmup", dsa_indexer_loss_coef=1.0
    ))
    # No router-load data is needed at all: the complete main model is frozen.
    trainer._update_ling_router_bias([{}], torch.tensor(1.0))


def test_dsa_warmup_does_not_load_unused_target_vocabulary_head(monkeypatch):
    from aspec.models.target import target_utils
    from aspec.training.dflash_trainer import DFlashTrainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = SimpleNamespace(config=config("indexer_warmup"))
    monkeypatch.setattr(target_utils, "load_synced_target_lm_head",
                        lambda *_a, **_kw: pytest.fail("Indexer warmup loaded a vocabulary head"))
    trainer._init_target_lm_head("/unused/target")
    assert trainer.target_lm_head is None and trainer.verifier_norm is None


def test_selected_teacher_uses_its_own_chunk_and_is_not_recomputed_in_backward(monkeypatch):
    import aspec.models.draft.dsa as module

    values = inputs()
    values["context_hidden"] = torch.randn(1, 128, 16, requires_grad=True)
    values["main_key"] = torch.randn(1, 132, 1, 4, requires_grad=True)
    values["layout"] = build_dsa_token_layout(
        values["metadata"], torch.arange(128)[None], values["draft_position_ids"],
        rope_dim=2, rope_theta=10000.0,
    )
    sizes = []
    original = module._selected_teacher_target

    def observe(**kwargs):
        sizes.append(kwargs["main_query"].shape[1])
        return original(**kwargs)

    monkeypatch.setattr(module, "_selected_teacher_target", observe)
    layer = DSALayer(config(dsa_workspace_bytes=2048), 0)
    _, loss, _ = layer(**values)
    assert sizes == [4]  # Full-context search needs four chunks; selected KL needs one.
    loss.sum().backward()
    assert sizes == [4]  # Detached K-sized teacher is retained; no repeated GEMM.
