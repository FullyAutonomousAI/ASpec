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

"""Independent QSA references for grouping, selection, teacher and lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import aspec.models.draft.qsa as qsa_module
from aspec.config.draft_attention import resolve_draft_attention
from aspec.config.train_config import load_config
from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from aspec.models.draft.qsa import (
    QSAForwardMetadata,
    QSAIndexer,
    QSALayer,
    _distributed_full_teacher_target,
    _full_teacher_target,
    _selected_teacher_target,
    build_qsa_compression_layout,
    build_qsa_reference_mask_mod,
    qsa_eligible_groups,
)
from aspec.planning import PlanCompileError, compile_execution_plan
from tools.convert_to_hf import _serving_config

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/examples/ling3_flash_qsa_dspark.yaml"
QSA_DRAFT_CONFIG = (
    ROOT / "aspec/config/dspark_ling3_flash_qsa_dense_draft_config.json"
)


def _config(stage="sparse", **changes):
    values = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "head_dim": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_target_layers": 1,
        "target_hidden_size": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "layer_types": ["qwen_sparse_attention"],
        "qsa_training_stage": stage,
        "qsa_token_budget": 4,
        "qsa_compress_ratio": 2,
        "qsa_index_n_heads": 2,
        "qsa_index_head_dim": 4,
        "qsa_rope_dim": 2,
        "qsa_workspace_bytes": 1 << 20,
    }
    values.update(changes)
    return DFlashConfig(**values)


def test_qsa_policy_roundtrip_and_invalid_contracts():
    config = _config()
    policy = resolve_draft_attention(config.to_dict())
    assert policy["context_windows"] == [None]
    assert policy["qsa"] == {
        "variant": "qsa",
        "training_stage": "sparse",
        "token_budget": 4,
        "compress_ratio": 2,
        "block_topk": 2,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "rope_dim": 2,
        "rope_theta": 10000.0,
        "indexer_loss_coef": 1.0,
        "workspace_bytes": 1 << 20,
        "indexer_input_detached": False,
        "cross_layer_reuse": False,
        "draft_block_index_reuse": True,
        "sparse_core_backend": "flash_attn_varlen_gather",
    }
    for change in (
        {"qsa_token_budget": 3},
        {"qsa_compress_ratio": 1},
        {"qsa_rope_dim": 3},
        {"qsa_rope_dim": 6},
        {"qsa_training_stage": "unknown"},
        {"qsa_training_stage": "indexer_warmup", "qsa_indexer_loss_coef": 0.0},
        {"layer_types": ["full_attention"], "qsa_token_budget": 4},
        {
            "layer_types": ["compressed_sparse_attention", "qwen_sparse_attention"],
            "num_hidden_layers": 2,
            "sliding_window": 4,
            "compress_ratios": [4, 0],
        },
    ):
        with pytest.raises(ValueError):
            DFlashConfig(**(_config().to_dict() | change))


def test_qsa_packed_groups_tail_and_visibility_do_not_cross_documents():
    docs = torch.tensor([[0] * 5 + [1] * 7])
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]])
    anchors = torch.tensor([[4, 8, 11]])
    metadata = QSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=1,
        context_doc_ids=docs,
        anchor_doc_ids=docs.gather(1, anchors),
    )
    layout = build_qsa_compression_layout(metadata, positions, ratio=4)
    assert layout.valid.tolist() == [[True, True, False]]
    assert layout.token_indices[0, :2].tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert layout.context_group_ids.tolist() == [[0, 0, 0, 0, -1, 1, 1, 1, 1, -1, -1, -1]]
    assert qsa_eligible_groups(metadata, layout).tolist() == [
        [[True, False, False], [False, False, False], [False, True, False]]
    ]


def _teacher_reference(query, key, visible, groups, scale):
    probabilities = torch.zeros(query.shape[0], query.shape[1], key.shape[1])
    repetitions = query.shape[2] // key.shape[2]
    for batch in range(query.shape[0]):
        for row in range(query.shape[1]):
            for head in range(query.shape[2]):
                scores = (
                    query[batch, row, head].float()
                    @ key[batch, :, head // repetitions].float().T
                ) * scale
                scores = scores.masked_fill(~visible[batch, row], -torch.inf)
                probabilities[batch, row] += scores.softmax(dim=-1)
    pooled = torch.stack(
        [probabilities[:, :, group].max(dim=-1).values for group in groups], dim=-1
    )
    return pooled / pooled.sum(dim=-1, keepdim=True)


def test_qsa_full_and_selected_teacher_match_scalar_head_sum_then_max_pool():
    torch.manual_seed(97)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[5]]),
        block_keep_mask=torch.tensor([[True]]),
        block_size=2,
    )
    positions = torch.arange(6).view(1, -1)
    layout = build_qsa_compression_layout(metadata, positions, ratio=2)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 8, 1, 4)
    scale = 0.5
    full = _full_teacher_target(
        main_query=query,
        main_key=key,
        metadata=metadata,
        layout=layout,
        start=0,
        end=2,
        scale=scale,
    )
    visible = torch.zeros(1, 2, 8, dtype=torch.bool)
    visible[:, :, :5] = True
    visible[:, :, 6:] = True
    expected = _teacher_reference(query, key, visible, [[0, 1], [2, 3]], scale)
    torch.testing.assert_close(full[:, :, :2], expected)
    assert torch.equal(full[:, :, 2], torch.zeros_like(full[:, :, 2]))

    selected = torch.tensor([[[0, 1], [0, 1]]])
    sparse = _selected_teacher_target(
        main_query=query,
        main_key=key,
        metadata=metadata,
        layout=layout,
        selected=selected,
        start=0,
        end=2,
        scale=scale,
    )
    # Selected groups 0/1, incomplete tail token 4 and this Draft block 6/7.
    compact_key = key[:, [0, 1, 2, 3, 4, 6, 7]]
    compact_visible = torch.ones(1, 2, 7, dtype=torch.bool)
    expected_sparse = _teacher_reference(
        query, compact_key, compact_visible, [[0, 1], [2, 3]], scale
    )
    torch.testing.assert_close(sparse, expected_sparse)


def test_qsa_indexer_projects_average_before_norm_and_partial_rope():
    torch.manual_seed(101)
    config = _config()
    indexer = QSAIndexer(config).float()
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[4]]),
        block_keep_mask=torch.tensor([[True]]),
        block_size=1,
    )
    hidden = torch.randn(1, 6, 16, requires_grad=True)
    draft = torch.randn(1, 1, 16, requires_grad=True)
    positions = torch.zeros(1, 6, dtype=torch.long)
    layout = build_qsa_compression_layout(metadata, positions, ratio=2)
    query, key = indexer._project(
        draft_hidden=draft,
        context_hidden=hidden,
        draft_position_ids=torch.zeros(1, 1, dtype=torch.long),
        layout=layout,
    )
    raw_key = torch.nn.functional.linear(hidden, indexer.k_proj.weight)
    expected_key = raw_key.view(1, 3, 2, 4).mean(dim=2)
    expected_key = expected_key * torch.rsqrt(
        expected_key.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expected_key = expected_key * indexer.k_norm.weight
    torch.testing.assert_close(key, expected_key)
    scores = indexer._scores(query, key)
    expected_scores = sum(
        torch.einsum("bqd,bkd->bqk", query[:, :, head], key).relu()
        for head in range(config.qsa_index_n_heads)
    ) * config.qsa_index_head_dim**-0.5
    torch.testing.assert_close(scores, expected_scores)
    scores.sum().backward()
    assert hidden.grad is not None and draft.grad is not None


def test_qsa_sparse_mask_selects_original_tokens_tail_and_current_draft_block():
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[5]]),
        block_keep_mask=torch.tensor([[True]]),
        block_size=2,
    )
    layout = build_qsa_compression_layout(metadata, torch.arange(6)[None], ratio=2)
    selected = torch.tensor([[[0], [1]]])
    mask = build_qsa_reference_mask_mod(
        metadata=metadata,
        layout=layout,
        selected=selected,
        context_length=6,
        draft_length=2,
        device=torch.device("cpu"),
    )
    assert [bool(mask(0, 0, 0, torch.tensor(k))) for k in range(8)] == [
        True, True, False, False, True, False, True, True
    ]
    assert [bool(mask(0, 0, 1, torch.tensor(k))) for k in range(8)] == [
        False, False, True, True, True, False, True, True
    ]


def test_qsa_keeps_one_target_aligned_selection_per_draft_block():
    torch.manual_seed(109)
    config = _config()
    layer = QSALayer(config, 0)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[3, 5]]),
        block_keep_mask=torch.tensor([[True, True]]),
        block_size=2,
    )
    context = torch.randn(1, 6, 16, requires_grad=True)
    draft = torch.randn(1, 4, 16, requires_grad=True)
    context_positions = torch.arange(6).view(1, -1)
    draft_positions = torch.tensor([[3, 4, 5, 6]])
    layout = build_qsa_compression_layout(metadata, context_positions, ratio=2)
    selected, loss, observation = layer(
        draft_hidden=draft,
        context_hidden=context,
        draft_position_ids=draft_positions,
        metadata=metadata,
        layout=layout,
        main_query=torch.randn(1, 4, 2, 4),
        main_key=torch.randn(1, 10, 1, 4),
        scale=0.5,
    )
    assert selected.shape[:2] == (1, 2)
    torch.testing.assert_close(loss[:, 0], loss[:, 1])
    torch.testing.assert_close(loss[:, 2], loss[:, 3])
    assert observation["draft_block_index_reuse"] is True
    loss.sum().backward()
    assert context.grad is not None and draft.grad is not None


def test_qsa_indexer_warmup_returns_only_block_repeated_kl():
    torch.manual_seed(113)
    config = _config("indexer_warmup")
    layer = QSALayer(config, 0)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[3, 5]]),
        block_keep_mask=torch.tensor([[True, True]]),
        block_size=2,
    )
    context = torch.randn(1, 6, 16, requires_grad=True)
    draft = torch.randn(1, 4, 16, requires_grad=True)
    layout = build_qsa_compression_layout(
        metadata, torch.arange(6).view(1, -1), ratio=2
    )
    selected, loss, observation = layer(
        draft_hidden=draft,
        context_hidden=context,
        draft_position_ids=torch.tensor([[3, 4, 5, 6]]),
        metadata=metadata,
        layout=layout,
        main_query=torch.randn(1, 4, 2, 4),
        main_key=torch.randn(1, 10, 1, 4),
        scale=0.5,
    )
    assert selected is None
    assert loss.shape == (1, 4)
    torch.testing.assert_close(loss[:, 0], loss[:, 1])
    torch.testing.assert_close(loss[:, 2], loss[:, 3])
    assert observation["training_stage"] == "indexer_warmup"
    loss.sum().backward()
    assert context.grad is not None and draft.grad is not None


def test_qsa_dense_stage_skips_compression_layout(monkeypatch):
    config = _config("dense")
    layer = QSALayer(config, 0)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[3]]),
        block_keep_mask=torch.tensor([[True]]),
        block_size=2,
    )
    selected, loss, observation = layer(
        draft_hidden=torch.randn(1, 2, 16),
        context_hidden=torch.randn(1, 4, 16),
        draft_position_ids=torch.tensor([[3, 4]]),
        metadata=metadata,
        layout=None,
        main_query=torch.randn(1, 2, 2, 4),
        main_key=torch.randn(1, 6, 1, 4),
        scale=0.5,
    )
    assert selected is None
    assert torch.equal(loss, torch.zeros_like(loss))
    assert observation["training_stage"] == "dense"

    def fail_if_built(*args, **kwargs):
        raise AssertionError("dense QSA must not build O(context) compression metadata")

    monkeypatch.setattr(qsa_module, "build_qsa_compression_layout", fail_if_built)
    model = DFlashDraftModel(config)
    monkeypatch.setattr(
        model.layers[0],
        "forward",
        lambda **kwargs: kwargs["draft_hidden"],
    )
    output = model(
        draft_input_ids=None,
        context_feature=torch.randn(1, 4, 16),
        draft_position_ids=torch.tensor([[3, 4]]),
        context_position_ids=torch.arange(4).view(1, -1),
        noise_embedding=torch.randn(1, 2, 16),
        qsa_metadata=metadata,
    )
    assert output.shape == (1, 2, 16)


def test_qsa_compression_layout_is_built_once_and_shared(monkeypatch):
    config = _config(
        "sparse",
        num_hidden_layers=2,
        layer_types=["qwen_sparse_attention"] * 2,
    )
    model = DFlashDraftModel(config)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[4, 8]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        block_size=2,
    )
    seen_layouts = []
    original = build_qsa_compression_layout
    build_calls = []

    def observe_layout(*args, **kwargs):
        build_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(qsa_module, "build_qsa_compression_layout", observe_layout)
    for layer in model.layers:
        monkeypatch.setattr(
            layer,
            "forward",
            lambda _seen=seen_layouts, **kwargs: (
                _seen.append(kwargs["qsa_layout"]) or kwargs["draft_hidden"]
            ),
        )
    output = model(
        draft_input_ids=None,
        context_feature=torch.randn(1, 12, 16),
        draft_position_ids=torch.tensor([[4, 5, 8, 9]]),
        context_position_ids=torch.arange(12).view(1, -1),
        noise_embedding=torch.randn(1, 4, 16),
        qsa_metadata=metadata,
    )

    assert output.shape == (1, 4, 16)
    assert len(build_calls) == 1
    assert len(seen_layouts) == 2 and seen_layouts[0] is seen_layouts[1]


@pytest.mark.parametrize(
    "stage,trainable_prefix",
    [("dense", "non_indexer"), ("indexer_warmup", "indexer"), ("sparse", "all")],
)
def test_qsa_phase_trainability(stage, trainable_prefix):
    model = DFlashDraftModel(_config(stage))
    indexer = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".qsa_layer.indexer." in name
    ]
    other = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".qsa_layer.indexer." not in name
    ]
    if trainable_prefix == "non_indexer":
        assert indexer and not any(indexer) and any(other)
    elif trainable_prefix == "indexer":
        assert indexer and all(indexer) and not any(other)
    else:
        assert all(indexer) and all(other)


def test_qsa_serving_export_fails_closed():
    with pytest.raises(NotImplementedError, match="QSA"):
        _serving_config(_config().to_dict())


def test_public_ling_qsa_recipe_has_three_topology_identical_stages(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = load_config(str(RECIPE), save_snapshot=False)
    plan = compile_execution_plan(config, config_path=str(RECIPE))
    args = plan.to_runtime_args()
    policy = plan.workload.models["draft"]["attention"]["qsa"]
    assert policy == {
        "variant": "qsa",
        "training_stage": "dense",
        "token_budget": 2048,
        "compress_ratio": 4,
        "block_topk": 512,
        "index_n_heads": 4,
        "index_head_dim": 128,
        "rope_dim": 64,
        "rope_theta": 6000000.0,
        "indexer_loss_coef": 1.0,
        "workspace_bytes": 128 * 1024 * 1024,
        "indexer_input_detached": False,
        "cross_layer_reuse": False,
        "draft_block_index_reuse": True,
        "sparse_core_backend": "flash_attn_varlen_gather",
    }
    capability = next(
        item for item in plan.capabilities if item.name == "draft_attention_execution"
    )
    assert capability.requirement == "required"
    assert capability.resolved["attention_policy"]["qsa"] == policy
    assert args.attention_backend == "flex_attention"
    assert args.draft_context_parallel_size == 1
    assert not args.compile_model and not args.dflash_packing
    assert args.fsdp_strategy == "REPLICATE"
    assert args.max_seq_length == 261888
    assert args.inference_num_gpus == 4
    assert args.inference_num_gpus_per_engine == 4
    assert args.sglang_tp_size == 4
    assert args.sglang_extra_args["max_total_tokens"] == 262144
    assert args.mooncake_local_buffer_size == "8GB"

    configs = []
    for stage in ("dense", "indexer", "sparse"):
        path = (
            ROOT
            / "aspec/config"
            / f"dspark_ling3_flash_qsa_{stage}_draft_config.json"
        )
        raw = json.loads(path.read_text())
        configs.append(raw)
    stages = [raw.pop("qsa_training_stage") for raw in configs]
    assert stages == ["dense", "indexer_warmup", "sparse"]
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("attention_backend", "cudnn_flex_attention", "requires.*flex_attention or fa4"),
        ("compile_model", True, "does not yet support compile_model"),
        ("dflash_packing", True, "packing requires.*fa4"),
    ],
)
def test_qsa_plan_fails_closed_outside_validated_execution(
    monkeypatch, field, value, error
):
    monkeypatch.chdir(ROOT)
    config = load_config(str(RECIPE), save_snapshot=False)
    setattr(config.training, field, value)
    with pytest.raises(PlanCompileError, match=error):
        compile_execution_plan(config, config_path=str(RECIPE))


def test_qsa_plan_accepts_true_thd_packing_at_cp1(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = load_config(str(RECIPE), save_snapshot=False)
    config.training.attention_backend = "fa4"
    config.training.dflash_packing = True
    plan = compile_execution_plan(config, config_path=str(RECIPE))
    args = plan.to_runtime_args()
    assert args.attention_backend == "fa4"
    assert args.draft_context_parallel_size == 1


def test_qsa_plan_accepts_true_thd_packing_with_direct_cp(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = load_config(str(RECIPE), save_snapshot=False)
    config.training.attention_backend = "fa4"
    config.training.draft_context_parallel_size = 2
    config.training.training_num_gpus_per_node = 2
    config.training.dflash_packing = True
    plan = compile_execution_plan(config, config_path=str(RECIPE))
    args = plan.to_runtime_args()
    assert args.attention_backend == "fa4"
    assert args.draft_context_parallel_size == 2
    assert args.draft_context_shard_alignment == 4
    assert plan.policy.capacity["backend_tokens"]["training_sharding"][
        "context_shard_alignment"
    ] == 4
    assert args.dflash_packing


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_qsa_plan_accepts_all_stages_with_true_thd_and_direct_cp(
    monkeypatch, tmp_path, stage, cp_size
):
    monkeypatch.chdir(ROOT)
    raw = json.loads(QSA_DRAFT_CONFIG.read_text())
    raw["qsa_training_stage"] = stage
    draft = tmp_path / f"qsa-thd-{stage}-cp{cp_size}.json"
    draft.write_text(json.dumps(raw))
    config = load_config(str(RECIPE), save_snapshot=False)
    config.model.draft_model_config = str(draft)
    config.training.attention_backend = "fa4"
    config.training.draft_context_parallel_size = cp_size
    config.training.training_num_gpus_per_node = cp_size
    config.training.dflash_packing = True
    # Retain the complete CP-sharded sample in the bounded source ring.
    config.mooncake.source_gpu_ring_slots = cp_size
    plan = compile_execution_plan(config, config_path=str(RECIPE))
    args = plan.to_runtime_args()
    assert args.draft_context_parallel_size == cp_size
    assert args.draft_context_shard_alignment == 4
    assert args.dflash_packing


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("anchors", [[1, 5], [4, 8]])
@pytest.mark.parametrize("reverse", [False, True])
def test_distributed_teacher_uses_same_complete_groups(ratio, anchors, reverse):
    torch.manual_seed(107)
    block_size = 2
    positions = torch.arange(8).view(1, -1)
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([anchors]),
        block_keep_mask=torch.tensor([[True, True]]),
        block_size=block_size,
    )
    layout = build_qsa_compression_layout(metadata, positions, ratio=ratio)
    rows = torch.tensor([3, 0]) if reverse else torch.arange(4)
    query = torch.randn(1, rows.numel(), 4, 8)
    key = torch.randn(1, 12, 2, 8)
    expected = _full_teacher_target(
        main_query=query, main_key=key, metadata=metadata, layout=layout,
        start=0, end=rows.numel(), scale=8**-0.5, query_indices=rows,
    )
    actual, valid = _distributed_full_teacher_target(
        main_query=query, main_key=key, metadata=metadata, layout=layout,
        query_indices=rows, scale=8**-0.5,
        direct_cp_metadata=SimpleNamespace(
            context_position_ids=positions, local_block_ids=torch.arange(2),
            context_length=8, cp_size=1, group=None,
        ),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(valid, expected.sum(-1) > 0)


def test_qsa_warmup_uneven_owners_agree_on_collective_chunks():
    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, head_dim=4,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        num_target_layers=1, target_hidden_size=16, vocab_size=32,
        max_position_embeddings=64, layer_types=["qwen_sparse_attention"],
        qsa_training_stage="indexer_warmup", qsa_token_budget=8,
        qsa_compress_ratio=4, qsa_index_n_heads=2,
        qsa_index_head_dim=4, qsa_rope_dim=2, qsa_workspace_bytes=1100,
    )
    schedules = []
    for context_length in [12, 16]:
        positions = torch.arange(context_length)[None]
        metadata = QSAForwardMetadata(
            anchor_positions=torch.tensor([[1, 9, 17, 25]]),
            block_keep_mask=torch.ones(1, 4, dtype=torch.bool), block_size=2,
            context_physical_position_ids=positions,
        )
        layout = build_qsa_compression_layout(metadata, positions, ratio=4)
        chunks = []
        def teacher(**kwargs):
            query = kwargs["main_query"]
            chunks.append(query.shape[1])
            return torch.zeros(1, query.shape[1], layout.num_groups), torch.ones(1, query.shape[1], dtype=torch.bool)
        direct = SimpleNamespace(cp_size=2, group=object(), sequence_length=28, layout="contiguous", queries_per_rank=4, _fa4_cache={})
        with (
            patch("aspec.models.ops.direct_cp.all_gather_query_rows", side_effect=lambda x, **kw: torch.cat((x, x), dim=1)),
            patch("aspec.models.ops.distributed_sparse.distributed_log_softmax", side_effect=lambda s, e, **kw: (s * 0, e.any(-1))),
            patch("aspec.models.draft.qsa._distributed_full_teacher_target", side_effect=teacher),
            patch("torch.distributed.nn.functional.all_reduce", side_effect=lambda x, **kw: x),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            QSAIndexer(config)(
                draft_hidden=torch.randn(1, 2, 16), context_hidden=torch.randn(1, context_length, 16),
                draft_position_ids=torch.tensor([[1, 9]]), metadata=metadata, layout=layout,
                main_query=torch.randn(1, 2, 4, 4), main_key=torch.randn(1, context_length + 4, 2, 4),
                stage="indexer_warmup", main_scale=0.5,
                query_indices=torch.arange(0, 8, 2), direct_cp_metadata=direct,
            )
        schedules.append(chunks)
    assert schedules[0] == schedules[1], schedules
