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

"""Native export preserves stages, tensor precision and real proposal semantics."""

import json

import pytest
import torch

from aspec.draft_artifact import BlockDraftConsumer, MLAEagleConsumer, load_native_draft
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.training.checkpoint import load_initial_draft_weights
from tools import convert_to_hf


@pytest.fixture(autouse=True)
def eager_rotary_for_unit_math(monkeypatch):
    # GPU gates exercise the real compiler. CPU units need the same math, not
    # a ten-second first-use Inductor compilation for an eight-dimensional RoPE.
    from aspec.models.draft import deepseek_eagle
    from aspec.models.draft.llama3_eagle import LlamaRotaryEmbedding

    function = deepseek_eagle._apply_rotary_pos_emb_interleaved
    monkeypatch.setattr(deepseek_eagle, "_apply_rotary_pos_emb_interleaved",
                        getattr(function, "_torchdynamo_orig_callable", function))
    function = LlamaRotaryEmbedding.forward
    monkeypatch.setattr(LlamaRotaryEmbedding, "forward",
                        getattr(function, "_torchdynamo_orig_callable", function))


def config(kind="csa", stage="sparse"):
    raw = dict(architectures=["Qwen3DSparkModel"], hidden_size=16, intermediate_size=32,
               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
               vocab_size=32, target_hidden_size=16, num_target_layers=2,
               num_aux_hidden_states=2, target_num_hidden_layers=6, target_layer_ids=[1, 4],
               max_position_embeddings=128, mask_token_id=31, markov_rank=4,
               enable_confidence_head=True, confidence_head_with_markov=True)
    if kind == "csa":
        raw.update(layer_types=["compressed_sparse_attention"], sliding_window=4,
                   compress_ratios=[4], csa_training_stage=stage, index_n_heads=2,
                   index_head_dim=8, index_topk=2)
    elif kind == "qsa":
        raw.update(layer_types=["qwen_sparse_attention"], qsa_training_stage=stage,
                   qsa_token_budget=4, qsa_compress_ratio=2, qsa_index_n_heads=2,
                   qsa_index_head_dim=8, qsa_rope_dim=4)
    elif kind == "dsa":
        raw.update(layer_types=["deepseek_sparse_attention"], dsa_training_stage=stage,
                   dsa_token_budget=3, dsa_index_n_heads=2, dsa_index_head_dim=8,
                   dsa_rope_dim=4, dsa_q_lora_rank=8)
    else:
        raw.update(architectures=["K3DSparkModel" if kind in {"mla", "mla_ling_moe"} else "Eagle3DeepseekV2ForCausalLM"],
                   num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8,
                   qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
                   pretraining_tp=1, pad_token_id=0, rope_theta=10000.0)
        if kind == "mla_ling_moe":
            raw.update(ffn_type="ling_moe", num_experts=4, num_experts_per_tok=2,
                       moe_intermediate_size=12, num_shared_experts=1,
                       moe_shared_expert_intermediate_size=12, n_group=2, topk_group=1,
                       moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001,
                       bias_zero_mean_update=True, router_aux_loss_coef=0,
                       router_z_loss_coef=0.0000035)
    return AutoDraftModelConfig.from_dict(raw)


@pytest.mark.parametrize("kind,stage", [("csa", "dense"), ("csa", "indexer_warmup"),
                                       ("csa", "sparse"), ("qsa", "dense"),
                                       ("qsa", "indexer_warmup"), ("qsa", "sparse"),
                                       ("dsa", "dense"), ("dsa", "indexer_warmup"), ("dsa", "sparse"),
                                       ("mla", "dense"), ("mla_ling_moe", "dense"),
                                       ("eagle_mla", "dense")])
def test_native_checkpoint_export_loads_every_tensor_and_semantics(tmp_path, monkeypatch, kind, stage):
    cfg = config(kind, stage)
    model = AutoEagle3DraftModel.from_config(cfg, torch_dtype=torch.bfloat16)
    if hasattr(model, "keep_csa_projection_precision"):
        model.keep_csa_projection_precision()
    # Retain values that a BF16 intermediate would irreversibly round.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype == torch.float32:
                parameter.add_(0.0001234)
    source = tmp_path / "iter_0000002"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").write_bytes(b"stub")
    (source / "draft_config.json").write_text(json.dumps(cfg.to_dict()))
    wrapped = {"model_state.model.draft_model." + k: v for k, v in model.state_dict().items()}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)
    output = tmp_path / "asset"
    manifest = convert_to_hf.convert_checkpoint(input_dir=source, output_dir=output,
                                                artifact_format="aspec")
    assert manifest["storage_format"] == "aspec_native"
    assert manifest["draft_structure"]["loader_contract"]["runtime"] == "aspec"
    loaded = load_native_draft(output)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert value.dtype == loaded.state_dict()[key].dtype
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    for key, value in model.named_buffers():
        actual = dict(loaded.named_buffers())[key]
        assert value.dtype == actual.dtype
        torch.testing.assert_close(value, actual, atol=0, rtol=0)
    warm = AutoEagle3DraftModel.from_config(cfg, torch_dtype=torch.bfloat16)
    load_initial_draft_weights(warm, str(output))
    for key, value in model.state_dict().items():
        assert warm.state_dict()[key].dtype == value.dtype
        torch.testing.assert_close(warm.state_dict()[key], value, atol=0, rtol=0)
    load_initial_draft_weights(loaded, str(output))
    if kind == "dsa":
        from aspec.draft_artifact import validate_native_asset

        other_stage = config("dsa", "dense" if stage != "dense" else "sparse")
        with pytest.raises(ValueError, match="DSA stage"):
            validate_native_asset(output, expected_config=other_stage)
        # Weights-only initialization is explicitly a new phase, not exact resume.
        transitioned = AutoEagle3DraftModel.from_config(other_stage, torch_dtype=torch.bfloat16)
        load_initial_draft_weights(transitioned, str(output))
        assert transitioned.config.dsa_training_stage == other_stage.dsa_training_stage
        changed_budget = config("dsa", stage)
        changed_budget.dsa_token_budget += 1
        with pytest.raises(ValueError, match="DSA stage/semantics"):
            validate_native_asset(output, expected_config=changed_budget, weights_only=True)
        manifest_path = output / "export_manifest.json"
        original_manifest = manifest_path.read_text()
        corrupted = json.loads(original_manifest)
        corrupted["draft_structure"]["attention"]["dsa"]["training_stage"] = "invalid"
        manifest_path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError, match="DSA policy"):
            load_native_draft(output)
        manifest_path.write_text(original_manifest)
    raw = json.loads((output / "config.json").read_text())
    raw["rope_theta"] = 7
    (output / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        load_native_draft(output)


@pytest.mark.parametrize("norm_output", [False, True])
def test_mla_native_cached_proposal_matches_full_prefix_decoder(norm_output):
    cfg = config("eagle_mla")
    cfg.norm_output = norm_output
    model = AutoEagle3DraftModel.from_config(cfg)
    model.eval()
    ids = torch.tensor([[2, 3, 5, 7]])
    features = torch.randn(1, 4, 32)
    actual = MLAEagleConsumer(model).propose(ids, features, num_tokens=4)
    recurrent = model.project_hidden_states(features)
    embeddings = model.embed_input_ids(ids)
    expected = []
    for depth in range(4):
        positions = torch.arange(recurrent.shape[1])[None]
        hidden = model.backbone(embeddings, recurrent, None, positions, use_cache=False)[0]
        logits = model.compute_logits(hidden[:, -1:])[:, 0]
        expected.append(logits)
        token = logits.argmax(-1)
        output = model.norm(hidden[:, -1:]) if norm_output else hidden[:, -1:]
        recurrent = torch.cat((recurrent, output), 1)
        embeddings = torch.cat((embeddings, model.embed_input_ids(token[:, None])), 1)
    torch.testing.assert_close(actual["logits"], torch.stack(expected, 1), atol=1e-5, rtol=1e-5)


def test_block_consumer_self_conditions_markov_without_future_ground_truth(monkeypatch):
    model = AutoEagle3DraftModel.from_config(config("mla"))
    consumer = BlockDraftConsumer(model, block_size=3)
    hidden = torch.randn(1, 3, 16)
    seen = {}
    def backbone(ids, features, mask, **kwargs):
        seen["ids"] = ids
        seen["features"] = features
        assert kwargs["injected_anchors"].item() == 3
        assert kwargs["packed_context_positions"].tolist() == [[0, 1, 2, 3]]
        return hidden, None, None, None
    monkeypatch.setattr(consumer.backbone, "_draft_backbone", backbone)
    ids, features = torch.tensor([[1, 2, 3, 4]]), torch.randn(1, 4, 32)
    head = torch.randn(32, 16)
    result = consumer.propose(ids, features, lm_head_weight=head)
    previous, expected = ids[:, -1], []
    for row in hidden[0]:
        previous = (row @ head.T + model.markov_head.compute_step_bias(previous)).argmax(-1)
        expected.append(previous)
    torch.testing.assert_close(result["token_ids"], torch.stack(expected, 1), atol=0, rtol=0)
    assert seen["features"] is features
    assert seen["ids"].shape == (1, 7)  # cheap IDs only; no padded feature allocation


def test_mla_serving_is_not_claimed_for_a_gqa_verifier_cache():
    with pytest.raises(NotImplementedError, match="--format aspec"):
        convert_to_hf._serving_config(config("eagle_mla").to_dict())


@pytest.mark.parametrize("algorithm", ["eagle3", "dflash2"])
def test_native_format_does_not_advertise_an_incompatible_proposal(tmp_path, monkeypatch, algorithm):
    from aspec.models.draft.dflash2 import DFlash2Config

    raw = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
               num_attention_heads=2, num_key_value_heads=1, head_dim=8,
               vocab_size=32, target_hidden_size=16, num_target_layers=2,
               num_aux_hidden_states=2, target_num_hidden_layers=6, target_layer_ids=[1, 4],
               max_position_embeddings=128, mask_token_id=31, pad_token_id=0)
    if algorithm == "eagle3":
        cfg = AutoDraftModelConfig.from_dict(dict(raw, architectures=["LlamaForCausalLMEagle3"]))
    else:
        cfg = DFlash2Config(**raw, conv_group_size=4, selector_rank=4, selector_top_k=4)
    model = AutoEagle3DraftModel.from_config(cfg)
    source = tmp_path / "iter_0000002"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").touch()
    (source / "draft_config.json").write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + key: value for key, value in model.state_dict().items()
    })
    manifest = convert_to_hf.convert_checkpoint(input_dir=source, output_dir=tmp_path / "asset",
                                                artifact_format="aspec")
    loader = manifest["draft_structure"]["loader_contract"]
    assert loader["proposal_entrypoint"] is None
    assert loader["borrowed_tensors"] == []
    assert "warmstart" in loader["scope"]
