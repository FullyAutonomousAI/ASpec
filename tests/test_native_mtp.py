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

"""Native MTP identity, teacher-forced offsets, frozen weights and asset roundtrip."""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from aspec.config.train_config import load_config
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel
from aspec.models.mtp import MTPModel, mtp_step_weights
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.training.checkpoint import load_initial_draft_weights
from aspec.training.mtp_trainer import MTPTrainer
from aspec.training.trainer_actor import _trainer_class_for_config
from tools import convert_to_hf

ROOT = Path(__file__).parents[1]


def native_config():
    return {
        "model_type": "qwen3_5_text", "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 32, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "rms_norm_eps": 1e-6, "attention_dropout": 0.0, "attention_bias": False,
        "hidden_act": "silu", "max_position_embeddings": 128, "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
                            "mrope_interleaved": True},
    }


def model():
    return NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))


def test_native_dispatch_and_public_plan_have_their_own_identity():
    config = load_config(str(ROOT / "configs/recipes/qwen3_5_9b_mtp.yaml"), save_snapshot=False)
    plan = compile_execution_plan(config)
    assert plan.workload.objective["family"] == "mtp"
    assert plan.workload.objective["frozen_shared"] == ["embedding", "lm_head"]
    assert set(plan.workload.features["tensors"]) == {"hidden_states", "input_ids"}
    assert plan.workload.features["tensors"]["hidden_states"]["value_stage"] == "post_norm"
    args = plan.to_runtime_args()
    assert args.freeze_lm_head and args.ttt_length == args.mtp_steps == 3
    cfg = AutoDraftModelConfig.from_dict({"architectures": ["NativeMTPDraftModel"],
                                         "native_config": native_config()})
    assert _trainer_class_for_config(cfg) is MTPTrainer
    assert isinstance(AutoEagle3DraftModel.from_config(cfg), NativeMTPDraftModel)
    torch.testing.assert_close(torch.tensor(args.ploss_weights), torch.tensor(mtp_step_weights(3, 0.6)))


@pytest.mark.parametrize("change", [
    {"mtp_steps": 0}, {"mtp_step_weight_beta": -1.0},
    {"mtp_step_weight_beta": float("nan")}, {"attention_backend": "flex_attention"},
    {"dflash_packing": True}, {"compile_model": True},
])
def test_unimplemented_or_invalid_native_mtp_requests_fail_before_launch(change):
    config = load_config(str(ROOT / "configs/recipes/qwen3_5_9b_mtp.yaml"), save_snapshot=False)
    for key, value in change.items():
        config.training[key] = value
    with pytest.raises((ValueError, PlanCompileError), match="MTP"):
        compile_execution_plan(config)


def test_native_ce_and_complete_gradients_match_explicit_teacher_forcing():
    torch.manual_seed(91)
    draft = model()
    reference = copy.deepcopy(draft)
    wrapper = MTPModel(draft, length=3, gradient_checkpointing=True)
    ids = torch.randint(0, 32, (1, 13))
    hidden = torch.randn(1, 13, 16)
    mask = torch.ones_like(ids)
    mask[:, :5] = 0
    result = wrapper(ids, hidden, mask, torch.ones_like(ids))
    state = hidden[:, :9]
    positions = torch.arange(9)[None]
    expected = []
    for depth in range(3):
        embeds = reference.pre_fc_norm_embedding(reference.embed_tokens(ids[:, depth + 1:depth + 10]))
        fused = reference.fc(torch.cat((embeds, reference.pre_fc_norm_hidden(state)), -1))
        rotary = reference.rotary_emb(fused, positions[None].expand(3, -1, -1))
        state = reference.norm(reference.midlayer(fused, position_embeddings=rotary,
                                                  attention_mask=None, position_ids=positions))
        labels = ids[:, depth + 2:depth + 11].clone()
        labels[mask[:, depth + 2:depth + 11] == 0] = -100
        expected.append(torch.nn.functional.cross_entropy(
            reference.lm_head(state).flatten(0, 1), labels.flatten(), ignore_index=-100))
    weights = mtp_step_weights(3, 0.6)
    sum(w * value for w, value in zip(weights, result[0], strict=True)).backward()
    sum(w * value for w, value in zip(weights, expected, strict=True)).backward()
    for actual, target in zip(result[1], expected, strict=True):
        torch.testing.assert_close(actual, target)
    for (name, actual), (_, target) in zip(draft.named_parameters(), reference.named_parameters(), strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, target.grad)
        else:
            assert actual.grad is target.grad is None, name


def test_short_and_unsupervised_documents_return_connected_finite_zero():
    draft = model()
    output = MTPModel(draft, length=3)(
        torch.ones(2, 4, dtype=torch.long), torch.randn(2, 4, 16),
        torch.zeros(2, 4), torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
    )
    sum(output[0]).backward()
    assert all(value == 0 for value in output[1])
    assert all(p.grad is not None for p in draft.parameters() if p.requires_grad)
    assert all(p.grad is None for p in draft.parameters() if not p.requires_grad)


def test_extract_native_tensors_and_strict_export_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(93)
    source = model()
    original = source.state_dict_for_serving()
    target = tmp_path / "target"
    target.mkdir()
    save_file(original, str(target / "model.safetensors"))
    extracted = model()
    extracted.initialize_native_weights(str(target))
    for key, value in source.state_dict().items():
        if key not in {"embed_tokens.weight", "lm_head.weight"}:
            torch.testing.assert_close(extracted.state_dict()[key], value, rtol=0, atol=0)
    root = tmp_path / "iter_0000017"
    (root / "model").mkdir(parents=True)
    (root / "model/.metadata").touch()
    (root / "draft_config.json").write_text(json.dumps(source.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + key: value for key, value in source.state_dict().items()
    })
    output = tmp_path / "export"
    manifest = convert_to_hf.convert_checkpoint(input_dir=root, output_dir=output)
    assert manifest["algorithm"] == "mtp"
    assert manifest["draft_structure"]["loader_contract"]["speculative_algorithm"] == "EAGLE"
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForCausalLMMTP"]
    tensors = load_file(output / "model.safetensors")
    assert set(tensors) == set(original)
    restored = model()
    load_initial_draft_weights(restored, str(output))
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_native_frozen_binding_rejects_different_same_shape_target(monkeypatch):
    import aspec.models.draft.mtp as native
    model = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    tensors = {"embed": model.embed_tokens.weight.detach().clone(),
               "head": model.lm_head.weight.detach().clone()}
    monkeypatch.setattr(native, "load_tensor_from_pretrained", lambda _path, key: tensors[key])
    model.validate_target_binding("target", embedding_key="embed", lm_head_key="head")
    tensors["head"][0, 0] += 1
    with pytest.raises(ValueError, match="frozen tensor differs"):
        model.validate_target_binding("target", embedding_key="embed", lm_head_key="head")


def test_native_stitch_replaces_only_mtp_and_rejects_wrong_frozen_head(tmp_path, monkeypatch):
    from tools.stitch_native_mtp import stitch_native_mtp
    native = native_config()
    cfg = NativeMTPConfig(native_config=native)
    model = NativeMTPDraftModel(cfg)
    target, checkpoint = tmp_path / "target", tmp_path / "iter_0000002"
    target.mkdir()
    raw = {"model_type": "qwen3_5", "text_config": native}
    (target / "config.json").write_text(json.dumps(raw))
    base = model.state_dict_for_serving()
    base["model.language_model.embed_tokens.weight"] = base.pop("model.embed_tokens.weight")
    base["model.language_model.norm.weight"] = torch.randn(native["hidden_size"])
    save_file(base, str(target / "model.safetensors"))
    with torch.no_grad():
        model.fc.weight.add_(1)
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").write_bytes(b"stub")
    (checkpoint / "draft_config.json").write_text(json.dumps(cfg.to_dict()))
    wrapped = {"model_state.model.draft_model." + k: v for k, v in model.state_dict().items()}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)
    asset = tmp_path / "asset"
    convert_to_hf.convert_checkpoint(input_dir=checkpoint, output_dir=asset)
    output = tmp_path / "stitched"
    result = stitch_native_mtp(target=target, draft=asset, output=output)
    assert len(result["replaced_tensors"]) == 15
    tensors = {}
    for filename in output.glob("*.safetensors"):
        loaded = load_file(filename)
        assert not tensors.keys() & loaded.keys()
        tensors.update(loaded)
    assert tensors.keys() == base.keys()
    expected = model.state_dict_for_serving()
    for name, value in tensors.items():
        torch.testing.assert_close(value, expected[name] if name.startswith("mtp.") else base[name], atol=0, rtol=0)
    base["lm_head.weight"] = base["lm_head.weight"] + 1
    save_file(base, str(target / "model.safetensors"))
    with pytest.raises(ValueError, match="shared embedding/head"):
        stitch_native_mtp(target=target, draft=asset, output=tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_native_target_binding_validates_full_sharded_values_not_owner_slices(monkeypatch):
    import aspec.models.draft.mtp as native
    model = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    values = {"embed_tokens.weight": model.embed_tokens.weight.detach().clone(),
              "lm_head.weight": model.lm_head.weight.detach().clone()}
    calls = []
    class Sharded:
        def __init__(self, name):
            self.name = name
        def full_tensor(self):
            calls.append(self.name)
            return values[self.name]
        def to_local(self):
            raise AssertionError("An owner slice is not the complete frozen Target tensor")
    monkeypatch.setattr(model, "state_dict", lambda: {k: Sharded(k) for k in values})
    monkeypatch.setattr(native, "load_tensor_from_pretrained", lambda _path, key: values[key])
    model.validate_target_binding("target", embedding_key="embed_tokens.weight", lm_head_key="lm_head.weight")
    assert set(calls) == set(values)


def test_native_warmstart_rejects_foreign_embedding_before_target_reset(tmp_path, monkeypatch):
    from argparse import Namespace
    from contextlib import nullcontext

    from aspec.training import eagle3_trainer as module

    draft = model().bfloat16()
    asset, target = tmp_path / "asset", tmp_path / "target"
    asset.mkdir()
    target.mkdir()
    saved = draft.state_dict_for_serving()
    save_file(saved, str(asset / "model.safetensors"))
    # Same architecture, same head, but a different frozen embedding.
    expected = {key: value.clone() for key, value in saved.items()
                if key in {"model.embed_tokens.weight", "lm_head.weight"}}
    expected["model.embed_tokens.weight"][0, 0] += 1
    save_file(expected, str(target / "model.safetensors"))
    trainer = object.__new__(MTPTrainer)
    trainer.args = Namespace(attention_backend="sdpa", initial_draft_model_path=str(asset),
                             freeze_lm_head=True, embedding_key="model.embed_tokens.weight",
                             lm_head_key="lm_head.weight")
    monkeypatch.setattr(trainer, "_get_init_weight_context_manager", lambda: nullcontext)
    monkeypatch.setattr(module.AutoEagle3DraftModel, "from_config", lambda *_a, **_kw: draft)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.dist, "barrier",
                        lambda **_kw: pytest.fail("Invalid MTP binding reached distributed init"))
    with pytest.raises(ValueError, match="frozen tensor differs.*embed_tokens"):
        trainer.init_model(draft.config, str(target))


@pytest.mark.parametrize("field,value", [
    ("rope_parameters", {"rope_type": "default", "rope_theta": 17}),
    ("norm_output", True), ("fc_norm", True), ("is_causal", False),
])
def test_native_derived_semantics_cannot_be_silently_overridden(field, value):
    with pytest.raises(ValueError, match="Native MTP"):
        NativeMTPConfig(native_config=native_config(), **{field: value})


def test_native_mtp_asset_does_not_claim_a_block_proposal_consumer(tmp_path, monkeypatch):
    from aspec.draft_artifact import load_native_draft

    draft = model()
    source = tmp_path / "iter_0000002"
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").touch()
    (source / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + key: value for key, value in draft.state_dict().items()
    })
    output = tmp_path / "native"
    manifest = convert_to_hf.convert_checkpoint(input_dir=source, output_dir=output,
                                                artifact_format="aspec")
    contract = manifest["draft_structure"]["loader_contract"]
    assert contract["proposal_entrypoint"] is None
    assert contract["borrowed_tensors"] == []
    loaded = load_native_draft(output)
    for name, value in draft.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
