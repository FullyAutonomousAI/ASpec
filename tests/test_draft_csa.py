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

"""Independent CSA references for compression, selection and lifecycle contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from aspec.config.draft_attention import resolve_draft_attention
from aspec.config.train_config import load_config
from aspec.models.draft.csa import (
    CSAForwardMetadata,
    CSALayer,
    CSALightningIndexer,
    CSAOverlappingCompressor,
    build_csa_compression_layout,
    csa_eligible_groups,
)
from aspec.models.draft.dflash import (
    DFlashAttention,
    DFlashConfig,
    DFlashDraftModel,
    DFlashRotaryEmbedding,
)
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.dspark import DSparkModel
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.training.checkpoint import (
    _logical_training_topology,
    _validate_exact_resume_topology,
)
from tools.convert_to_hf import _serving_config

_ROOT = Path(__file__).parents[1]
_REFERENCE_CONFIG = _ROOT / "configs" / "examples" / "sglang_qwen3_8b_dspark.yaml"
_DRAFT_CONFIG = _ROOT / "aspec" / "config" / "dspark_draft_config.json"


def _draft_config(stage="dense"):
    return DFlashConfig(
        hidden_size=16,
        intermediate_size=32,
        vocab_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=1,
        target_hidden_size=16,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=["compressed_sparse_attention"],
        compress_ratios=[4],
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        csa_training_stage=stage,
        compress_rope_theta=40000.0,
    )


def test_csa_policy_roundtrip_and_invalid_semantics_fail_closed():
    config = _draft_config("sparse")
    restored = DFlashConfig.from_dict(config.to_dict())
    policy = resolve_draft_attention(restored.to_dict())
    assert policy["layer_types"] == ["compressed_sparse_attention"]
    assert policy["compress_ratios"] == [4]
    assert policy["csa"] == {
        "variant": "csa",
        "training_stage": "sparse",
        "compress_ratios": [4],
        "index_topk": 2,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "indexer_loss_coef": 1.0,
        "workspace_bytes": 128 * 1024 * 1024,
        "compress_rope_theta": 40000.0,
        "indexer_input_detached": True,
        "cross_layer_reuse": False,
    }
    for change in (
        {"compress_ratios": [0]},
        {"compress_ratios": [8]},
        {"csa_training_stage": "unknown"},
        {"csa_training_stage": "indexer_warmup", "csa_indexer_loss_coef": 0.0},
        {"layer_types": ["full_attention"], "compress_ratios": [4]},
    ):
        raw = config.to_dict() | change
        with pytest.raises(ValueError):
            resolve_draft_attention(raw)


def test_packed_compression_groups_and_visibility_never_cross_documents():
    docs = torch.tensor([[0] * 6 + [1] * 10])
    anchors = torch.tensor([[5, 8, 13]])
    metadata = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=1,
        context_window=4,
        context_doc_ids=docs,
        anchor_doc_ids=docs.gather(1, anchors),
    )
    document_positions = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    layout = build_csa_compression_layout(metadata, document_positions, ratio=4)

    assert layout.valid.tolist() == [[True, True, True, False]]
    assert layout.token_indices[0, :3].tolist() == [
        [0, 1, 2, 3],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
    ]
    assert layout.doc_ids.tolist() == [[0, 1, 1, -1]]
    assert csa_eligible_groups(metadata, layout).tolist() == [
        [[True, False, False, False], [False, False, False, False], [False, True, False, False]]
    ]


def test_overlap_compressor_matches_independent_scalar_definition():
    torch.manual_seed(71)
    compressor = CSAOverlappingCompressor(
        hidden_size=8,
        num_heads=1,
        head_dim=4,
        ratio=4,
        rope_theta=40000.0,
        norm_eps=1e-6,
    )
    hidden = torch.randn(1, 12, 8, requires_grad=True)
    metadata = CSAForwardMetadata(
        anchor_positions=torch.tensor([[8]]),
        block_keep_mask=torch.tensor([[True]]),
        block_size=1,
        context_window=4,
    )
    layout = build_csa_compression_layout(metadata, torch.zeros(1, 12, dtype=torch.long), 4)
    actual, rotated = compressor(hidden, layout)

    values = F.linear(hidden.float(), compressor.kv_proj.weight).view(1, 3, 4, 2, 1, 4)
    logits = F.linear(hidden.float(), compressor.gate_proj.weight).view(1, 3, 4, 2, 1, 4)
    logits = logits + compressor.position_bias.view(1, 1, 4, 2, 1, 4)
    expected_groups = []
    for group in range(3):
        group_values = values[:, group, :, 1]
        group_logits = logits[:, group, :, 1]
        if group:
            group_values = torch.cat((values[:, group - 1, :, 0], group_values), dim=1)
            group_logits = torch.cat((logits[:, group - 1, :, 0], group_logits), dim=1)
        expected_groups.append(
            (group_values * group_logits.softmax(dim=1)).sum(dim=1)
        )
    expected = torch.stack(expected_groups, dim=1)
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * compressor.norm.weight
    torch.testing.assert_close(actual, expected)
    # Zero RoPE positions make rotation the identity.
    torch.testing.assert_close(rotated, expected)
    actual.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert compressor.kv_proj.weight.grad is not None
    assert compressor.gate_proj.weight.grad is not None


def test_dflash_rope_cache_is_lazy_and_shape_bounded():
    rope = DFlashRotaryEmbedding(
        dim=8,
        max_position_embeddings=262_144,
        base=10_000.0,
    )
    assert rope.cos_cached.numel() == rope.sin_cached.numel() == 0

    cos, sin = rope(torch.empty(1), seq_len=16)

    assert cos.shape == sin.shape == (1, 1, 16, 8)
    assert rope.max_seq_len_cached == 36
    assert rope.cos_cached.shape[-2] == 36


def test_csa_compression_layout_is_built_once_per_ratio_and_shared(monkeypatch):
    config = DFlashConfig(
        **(
            _draft_config("sparse").to_dict()
            | {
                "num_hidden_layers": 2,
                "layer_types": ["compressed_sparse_attention"] * 2,
                "compress_ratios": [4, 4],
            }
        )
    )
    model = DFlashDraftModel(config)
    metadata = CSAForwardMetadata(
        anchor_positions=torch.tensor([[4, 8]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
        block_size=2,
        context_window=4,
    )
    seen_layouts = []
    original = build_csa_compression_layout
    with mock.patch(
        "aspec.models.draft.csa.build_csa_compression_layout",
        wraps=original,
    ) as build_layout:
        for layer in model.layers:
            monkeypatch.setattr(
                layer,
                "forward",
                lambda _seen=seen_layouts, **kwargs: (
                    _seen.append(kwargs["csa_prepared"][4][0])
                    or kwargs["draft_hidden"]
                ),
            )
        model(
            draft_input_ids=None,
            context_feature=torch.randn(1, 12, 16),
            draft_position_ids=torch.tensor([[4, 5, 8, 9]]),
            context_position_ids=torch.arange(12).view(1, -1),
            noise_embedding=torch.randn(1, 4, 16),
            csa_metadata=metadata,
        )

    assert build_layout.call_count == 1
    assert len(seen_layouts) == 2 and seen_layouts[0] is seen_layouts[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_zero_compressed_groups_still_enforce_csa_local_window():
    raw = _draft_config("dense").to_dict() | {
        "hidden_size": 32,
        "intermediate_size": 64,
        "head_dim": 16,
        "target_hidden_size": 32,
        "sliding_window": 1,
    }
    attention = DFlashAttention(DFlashConfig(**raw), layer_idx=0).cuda().float().eval()
    draft = torch.randn(1, 1, 32, device="cuda")
    context = torch.randn(1, 3, 32, device="cuda")
    metadata = CSAForwardMetadata(
        anchor_positions=torch.tensor([[2]], device="cuda"),
        block_keep_mask=torch.tensor([[True]], device="cuda"),
        block_size=1,
        context_window=1,
    )
    kwargs = {
        "draft_hidden": draft,
        "draft_position_ids": torch.tensor([[2]], device="cuda"),
        "context_position_ids": torch.arange(3, device="cuda").view(1, -1),
        "csa_metadata": metadata,
    }
    with torch.no_grad():
        reference = attention(context_hidden=context, **kwargs)
        changed = context.clone()
        changed[:, 0].add_(1000)
        actual = attention(context_hidden=changed, **kwargs)

    # Context token 0 is outside [anchor-window, anchor), and there are no
    # ratio-4 compressed groups. It must remain completely invisible.
    torch.testing.assert_close(actual, reference)


def test_dspark_indexer_warmup_skips_vocab_and_verifier_objectives():
    raw = _draft_config("indexer_warmup").to_dict() | {
        "markov_rank": 0,
        "enable_confidence_head": False,
        "confidence_head_with_markov": False,
    }
    draft = DSparkDraftModel(DSparkConfig(**raw)).float()
    model = DSparkModel(
        draft_model=draft,
        block_size=2,
        num_anchors=2,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )
    indexer_kl = torch.tensor([[0.2, 0.4, 0.6, 0.8]], requires_grad=True)

    def fake_backbone(*args, csa_losses_out, **kwargs):
        del args, kwargs
        csa_losses_out.append(indexer_kl)
        return (
            torch.zeros(1, 4, 16),
            torch.tensor([[3, 7]]),
            torch.tensor([[True, True]]),
            2,
        )

    with (
        mock.patch.object(model, "_draft_backbone", side_effect=fake_backbone),
        mock.patch.object(F, "linear", side_effect=AssertionError("vocab projection executed")),
    ):
        result = model(
            input_ids=torch.arange(12).view(1, 12).remainder(64),
            hidden_states_list=[torch.zeros(1, 12, 16)],
            loss_mask=torch.ones(1, 12),
            lm_head_weight=torch.empty(64, 16),
            last_hidden_states=None,
        )

    loss, accuracy, loss_pp, acc_pp, count_pp, components, loss_terms = result
    assert set(components) == {"csa_indexer_loss"}
    assert not model.uses_target_hidden_states
    assert accuracy.item() == 0
    assert torch.count_nonzero(loss_pp) == 0
    assert torch.count_nonzero(acc_pp) == 0
    assert torch.count_nonzero(count_pp) == 0
    torch.testing.assert_close(loss, loss_terms[0] / loss_terms[1])
    loss_terms[0].backward()
    assert indexer_kl.grad is not None and torch.count_nonzero(indexer_kl.grad) == 4


def test_indexer_warmup_metrics_report_only_indexer_objective():
    from aspec.training.dflash_trainer import DFlashTrainer

    trainer = object.__new__(DFlashTrainer)
    trainer.draft_model = SimpleNamespace(
        config=SimpleNamespace(
            csa_training_stage="indexer_warmup",
            csa_indexer_loss_coef=2.0,
        )
    )
    trainer.global_step = 3
    trainer.optimizer = SimpleNamespace(get_learning_rate=lambda: 1e-4)
    step = {
        "csa_indexer_loss": (torch.tensor(6.0), torch.tensor(3.0)),
    }
    with mock.patch("aspec.training.dflash_trainer.dist.get_rank", return_value=0):
        train = trainer._aggregate_metrics([step], step=4, grad_norm=torch.tensor(1.5))
        evaluation = trainer._aggregate_eval_metrics([step])

    assert train["train/csa_indexer_loss"] == 2.0
    assert train["train/avg_loss"] == 4.0
    assert "train/avg_acc" not in train and "train/simulated_acc_len" not in train
    assert evaluation == {"eval/avg_loss": 4.0, "eval/csa_indexer_loss": 2.0}


@pytest.mark.parametrize("stage", ["indexer_warmup", "sparse"])
def test_indexer_kl_detaches_main_inputs_but_trains_indexer(stage):
    torch.manual_seed(73)
    config = _draft_config(stage)
    layer = CSALayer(config, 0)
    anchors = torch.tensor([[8, 14]])
    metadata = CSAForwardMetadata(
        anchor_positions=anchors,
        block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        block_size=2,
        context_window=4,
    )
    draft = torch.randn(1, 4, 16, requires_grad=True)
    context = torch.randn(1, 16, 16, requires_grad=True)
    query = torch.randn(1, 4, 2, 8, requires_grad=True)
    compressed_key, compressed_value, _, per_query_loss, _ = layer(
        draft_hidden=draft,
        context_hidden=context,
        draft_position_ids=metadata.query_anchors,
        context_position_ids=torch.arange(16)[None],
        metadata=metadata,
        main_query=query,
        scale=8**-0.5,
    )
    per_query_loss.sum().backward()

    assert draft.grad is None and context.grad is None and query.grad is None
    assert any(parameter.grad is not None for parameter in layer.indexer.parameters())
    assert all(parameter.grad is None for parameter in layer.compressor.parameters())
    assert compressed_key.data_ptr() != compressed_value.data_ptr()


def test_sparse_selected_set_kl_matches_independent_scalar_definition():
    torch.manual_seed(79)
    indexer = CSALightningIndexer(_draft_config("sparse"), 0)
    index_scores = torch.randn(1, 5, 4, requires_grad=True)
    selected = torch.tensor(
        [[[0, 2], [3, 1], [2, -1], [-1, -1], [1, 0]]]
    )
    query = torch.randn(1, 5, 2, 8)
    compressed_key = torch.randn(1, 4, 1, 8)
    actual = indexer.loss_per_query(
        index_scores=index_scores,
        topk_indices=selected,
        main_query=query,
        main_compressed_key=compressed_key,
        eligible=torch.ones(1, 5, 4, dtype=torch.bool),
        stage="sparse",
        scale=8**-0.5,
    )

    expected = torch.zeros_like(actual)
    for q_idx in range(query.shape[1]):
        groups = [int(value) for value in selected[0, q_idx] if value >= 0]
        if not groups:
            continue
        head_probabilities = []
        for head in range(query.shape[2]):
            logits = torch.stack(
                [
                    torch.dot(query[0, q_idx, head], compressed_key[0, group, 0])
                    * (8**-0.5)
                    for group in groups
                ]
            )
            head_probabilities.append(logits.softmax(dim=-1))
        target = torch.stack(head_probabilities).sum(dim=0)
        target = target / target.sum()
        index_log_prob = index_scores[0, q_idx, groups].log_softmax(dim=-1)
        expected[0, q_idx] = (
            target * (target.clamp_min(1e-12).log() - index_log_prob)
        ).sum()

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert index_scores.grad is not None


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
def test_dspark_phase_freeze_policy_includes_late_created_heads(stage):
    raw = _draft_config(stage).to_dict() | {
        "markov_rank": 4,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    model = DSparkDraftModel(DSparkConfig(**raw))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if stage == "dense":
        assert trainable and not any("csa_layer.indexer" in name for name in trainable)
    elif stage == "indexer_warmup":
        assert trainable and all("csa_layer.indexer" in name for name in trainable)
    else:
        assert any("csa_layer.indexer" in name for name in trainable)
        assert any(name.startswith("markov_head") for name in trainable)


def test_csa_stage_is_exact_resume_topology_and_serving_fails_closed():
    def args(stage):
        return SimpleNamespace(world_size=1, draft_model_config_obj=_draft_config(stage))

    metadata = {
        "world_size": 1,
        "training_topology": _logical_training_topology(args("dense"), 1),
    }
    _validate_exact_resume_topology(args("dense"), metadata)
    with pytest.raises(RuntimeError, match="topology"):
        _validate_exact_resume_topology(args("indexer_warmup"), metadata)
    with pytest.raises(RuntimeError, match="CSA policy"):
        _validate_exact_resume_topology(args("dense"), {"world_size": 1})
    with pytest.raises(NotImplementedError, match="CSA"):
        _serving_config(_draft_config("sparse").to_dict())


@pytest.mark.parametrize(
    ("backend", "cp_size", "compile_model", "packing", "error"),
    [
        (
            "cudnn_flex_attention",
            1,
            False,
            False,
            "requires training.attention_backend=flex_attention or fa4",
        ),
        ("flex_attention", 1, True, False, "do not yet support compile_model"),
        (
            "flex_attention",
            1,
            False,
            True,
            "packing requires.*fa4",
        ),
    ],
)
def test_plan_rejects_unverified_csa_execution(
    tmp_path, backend, cp_size, compile_model, packing, error
):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        sliding_window=128,
        layer_types=["compressed_sparse_attention"] * raw["num_hidden_layers"],
        compress_ratios=[4] * raw["num_hidden_layers"],
        csa_training_stage="dense",
    )
    draft = tmp_path / "csa.json"
    draft.write_text(json.dumps(raw))
    config = load_config(
        config_path=str(_REFERENCE_CONFIG),
        cli_args=[
            f"model.draft_model_config={draft}",
            f"training.attention_backend={backend}",
        ],
    )
    config.training.draft_context_parallel_size = cp_size
    config.training.compile_model = compile_model
    config.training.dflash_packing = packing
    with pytest.raises(PlanCompileError, match=error):
        compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))


def test_plan_accepts_csa_true_thd_packing_at_cp1(tmp_path):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        sliding_window=128,
        layer_types=["compressed_sparse_attention"] * raw["num_hidden_layers"],
        compress_ratios=[4] * raw["num_hidden_layers"],
        csa_training_stage="sparse",
    )
    draft = tmp_path / "csa-thd.json"
    draft.write_text(json.dumps(raw))
    config = load_config(
        config_path=str(_REFERENCE_CONFIG),
        cli_args=[
            f"model.draft_model_config={draft}",
            "training.attention_backend=fa4",
        ],
    )
    config.training.dflash_packing = True
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    args = plan.to_runtime_args()
    assert args.attention_backend == "fa4"
    assert args.draft_context_parallel_size == 1
    assert args.dflash_packing


@pytest.mark.parametrize("stage", ["dense", "indexer_warmup", "sparse"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_plan_accepts_csa_true_thd_packing_with_direct_cp(
    tmp_path, stage, cp_size
):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        sliding_window=128,
        layer_types=["compressed_sparse_attention"] * raw["num_hidden_layers"],
        compress_ratios=[4] * raw["num_hidden_layers"],
        csa_training_stage=stage,
    )
    draft = tmp_path / f"csa-thd-{stage}-cp{cp_size}.json"
    draft.write_text(json.dumps(raw))
    config = load_config(
        config_path=str(_REFERENCE_CONFIG),
        cli_args=[
            f"model.draft_model_config={draft}",
            "training.attention_backend=fa4",
        ],
    )
    config.training.draft_context_parallel_size = cp_size
    config.training.training_num_gpus_per_node = cp_size
    config.training.dflash_packing = True
    plan = compile_execution_plan(config, config_path=str(_REFERENCE_CONFIG))
    args = plan.to_runtime_args()
    assert args.draft_context_parallel_size == cp_size
    assert args.draft_context_shard_alignment == 4
    assert args.dflash_packing


def test_plan_exposes_canonical_csa_policy(tmp_path):
    raw = json.loads(_DRAFT_CONFIG.read_text())
    raw.update(
        sliding_window=128,
        layer_types=["compressed_sparse_attention"] * raw["num_hidden_layers"],
        compress_ratios=[4] * raw["num_hidden_layers"],
        csa_training_stage="dense",
    )
    draft = tmp_path / "csa.json"
    draft.write_text(json.dumps(raw))
    overrides = [
        f"model.draft_model_config={draft}",
        "training.attention_backend=flex_attention",
    ]
    plan = compile_execution_plan(
        load_config(config_path=str(_REFERENCE_CONFIG), cli_args=overrides),
        config_path=str(_REFERENCE_CONFIG),
        cli_args=overrides,
    )
    policy = plan.workload.models["draft"]["attention"]
    assert policy["csa"]["training_stage"] == "dense"
    capability = next(
        item for item in plan.capabilities if item.name == "draft_attention_execution"
    )
    assert capability.requirement == "required"
    assert capability.resolved["attention_policy"] == policy


def test_public_ling_csa_recipe_has_three_topology_identical_stages(monkeypatch):
    monkeypatch.chdir(_ROOT)
    recipe = _ROOT / "configs" / "examples" / "ling3_flash_csa_dspark.yaml"
    plan = compile_execution_plan(
        load_config(str(recipe), save_snapshot=False), config_path=str(recipe)
    )
    args = plan.to_runtime_args()
    assert args.attention_backend == "flex_attention"
    assert args.draft_context_parallel_size == 1
    assert not args.compile_model and not args.dflash_packing
    assert args.fsdp_strategy == "REPLICATE"
    assert plan.workload.models["target"]["family"] == "ling3"
    assert plan.workload.models["draft"]["attention"]["csa"]["training_stage"] == "dense"

    configs = []
    for stage in ("dense", "indexer", "sparse"):
        path = _ROOT / "aspec" / "config" / f"dspark_ling3_flash_csa_{stage}_draft_config.json"
        raw = json.loads(path.read_text())
        configs.append(raw)
    stages = [raw.pop("csa_training_stage") for raw in configs]
    assert stages == ["dense", "indexer_warmup", "sparse"]
    assert configs[0] == configs[1] == configs[2]


def test_csa_runtime_status_reports_observed_backend_and_cp():
    from aspec.training.dflash_trainer import DFlashTrainer
    from aspec.training.trainer import Trainer

    trainer = object.__new__(DFlashTrainer)
    trainer.args = SimpleNamespace(
        attention_backend="flex_attention", draft_context_parallel_size=1
    )
    trainer.draft_model = SimpleNamespace(layers=[])
    trainer.dflash = SimpleNamespace(
        attention_policy_observed={
            "context_windows": [128],
            "window_semantics": "anchor_context",
            "draft_block_attention": "bidirectional",
            "csa": {"training_stage": "dense"},
            "layers": [{"layer": 0, "variant": "csa"}],
        }
    )
    with mock.patch.object(
        Trainer,
        "get_feature_contract_status",
        return_value={
            "state": "validated",
            "execution_metadata": {"feature_affinity": ["any"]},
        },
    ):
        status = trainer.get_feature_contract_status()

    assert status["execution_metadata"] == {
        "feature_affinity": ["any"],
        "draft_attention_backend": ["flex_attention"],
        "draft_context_parallel_size": [1],
    }
    assert status["draft_attention"]["forward_observed"] is True
