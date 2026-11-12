# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DMTP timing, known-token loss exclusion, and frozen-feature boundaries."""

import copy
import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aspec.models.dflash import DFlashModel, _create_dflash_mask_mod
from aspec.models.dmtp import (
    MTPAnchorFusion,
    MTPPrefix,
    frozen_mtp_predictions,
)
from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


def native_config():
    return {"model_type": "qwen3_5_text", "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "vocab_size": 32, "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rms_norm_eps": 1e-6, "mtp_num_hidden_layers": 1,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
                                "mrope_interleaved": True}}


def wrapper(use_hidden=True, algorithm="dflash"):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    cfg_cls, model_cls, root_cls = ((DFlashConfig, DFlashDraftModel, DFlashModel)
                                  if algorithm == "dflash" else
                                  (DSparkConfig, DSparkDraftModel, DSparkModel))
    config = cfg_cls(architectures=["DFlashDraftModel" if algorithm == "dflash" else "DSparkDraftModel"],
                         hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         vocab_size=32, num_target_layers=1, target_hidden_size=16,
                         target_num_hidden_layers=4, target_layer_ids=[1],
                         mask_token_id=31, max_position_embeddings=128,
                         mtp_conditioning={"mode": "hidden" if use_hidden else "token",
                                           "native_config": native_config()})
    draft = model_cls(config)
    draft.embed_tokens.requires_grad_(False)
    return root_cls(draft, block_size=5, num_anchors=2, attention_backend="sdpa")


def _nonzero_model(algorithm):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model = wrapper(algorithm=algorithm)
        nn.init.normal_(model.draft_model.mtp_fusion.proj.weight, std=0.1)
    return model


def _fusion_inputs():
    generator = torch.Generator().manual_seed(31)
    ids = torch.arange(12)[None]
    return dict(input_ids=ids, hidden_states_list=torch.randn(1, 12, 16, generator=generator),
                loss_mask=torch.ones_like(ids), lm_head_weight=torch.randn(32, 16, generator=generator),
                last_hidden_states=torch.randn(1, 12, 16, generator=generator),
                injected_anchors=torch.tensor([[3, 5]]),
                injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
                draft_prefix=MTPPrefix(torch.randn(1, 11, 16, generator=generator, requires_grad=True)))


def _reference_noise(self, embeddings, input_ids, anchors, keep, prefix):
    """Scalar anchor indexing and explicit RMS/linear equations, independent of fusion helpers."""
    draft = self.draft_model
    fusion = draft.mtp_fusion

    def normalize(value, norm):
        value = value.float()
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight

    blocks = embeddings.reshape(1, anchors.shape[1], self.block_size, -1)
    result = []
    for index, anchor in enumerate(anchors[0].tolist()):
        token = input_ids[:, anchor + 1] if keep[0, index] else input_ids.new_tensor([draft.mask_token_id])
        first = F.embedding(token, draft.embed_tokens.weight)
        hidden = (prefix.prediction_hidden[:, anchor - 1].detach() if keep[0, index]
                  else torch.zeros_like(first))
        features = torch.cat((normalize(first, fusion.embedding_norm),
                              normalize(hidden, fusion.hidden_norm)), dim=-1)
        first = first + F.linear(features, fusion.proj.weight)
        result.append(torch.cat((blocks[:, index, :1], first[:, None], blocks[:, index, 2:]), dim=1))
    return torch.cat(result, dim=1)


def _reference_logits_and_loss(model, hidden, inputs):
    """Dense equations for the conditioned CE/L1/confidence objective on complete blocks."""
    draft = model.draft_model
    hidden = hidden.reshape(1, 2, model.block_size, -1)
    offsets = torch.arange(model.block_size)[None, None]
    positions = inputs["injected_anchors"][..., None] + offsets
    is_dspark = hasattr(draft, "markov_head")
    positions = positions + int(is_dspark)
    labels = inputs["input_ids"][:, positions[0]]
    logits = F.linear(hidden, inputs["lm_head_weight"])
    previous = None
    if is_dspark:
        previous = F.embedding(inputs["input_ids"][:, positions[0] - 1],
                               draft.markov_head.markov_w1.weight)
        logits = logits + F.linear(previous, draft.markov_head.markov_w2.weight)
    ce = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none").view_as(labels)
    terms = model.ce_loss_alpha * ce
    if is_dspark:
        teacher = inputs["last_hidden_states"][:, positions[0] - 1]
        target = F.linear(teacher, inputs["lm_head_weight"]).float().softmax(-1)
        l1 = (logits.float().softmax(-1) - target).abs().sum(-1)
        confidence = F.linear(torch.cat((hidden, previous), dim=-1),
                              draft.confidence_head.proj.weight, draft.confidence_head.proj.bias).squeeze(-1)
        accept = (1 - 0.5 * l1).clamp(0, 1).detach()
        terms = terms + model.l1_loss_alpha * l1 + model.confidence_head_alpha * F.binary_cross_entropy_with_logits(
            confidence, accept, reduction="none")
    weights = (offsets >= model.prediction_start) * torch.exp(
        -(offsets - model.prediction_start).clamp_min(0).float() / model.loss_decay_gamma)
    weights = weights.expand_as(terms)
    return logits, (terms * weights).sum() / weights.sum()


def _assert_parameter_gradients(actual, expected, *, exact=False):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in actual_parameters.items():
        reference = expected_parameters[name]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference.grad,
                                       rtol=0 if exact else 2e-5, atol=0 if exact else 2e-6, msg=name)
        if not parameter.requires_grad:
            assert parameter.grad is None, name


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_nonzero_fusion_logits_loss_and_gradients_match_independent_reference(algorithm):
    actual = _nonzero_model(algorithm)
    reference = copy.deepcopy(actual)
    reference._condition_noise_embed = MethodType(_reference_noise, reference)
    inputs, reference_inputs = _fusion_inputs(), _fusion_inputs()
    captured = []
    handle = actual.draft_model.register_forward_hook(lambda _, args, output: captured.append(output))
    result = actual(**inputs)
    handle.remove()
    expected_hidden, _, _, _ = reference._draft_backbone(
        reference_inputs["input_ids"], reference_inputs["hidden_states_list"], reference_inputs["loss_mask"],
        injected_anchors=reference_inputs["injected_anchors"],
        injected_keep_mask=reference_inputs["injected_keep_mask"], draft_prefix=reference_inputs["draft_prefix"])
    logits, expected_loss = _reference_logits_and_loss(reference, expected_hidden, reference_inputs)
    actual_logits, _ = _reference_logits_and_loss(actual, captured[0], inputs)
    torch.testing.assert_close(actual_logits, logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result[0], expected_loss, rtol=2e-5, atol=2e-6)
    result[0].backward()
    expected_loss.backward()
    _assert_parameter_gradients(actual, reference)
    assert inputs["draft_prefix"].prediction_hidden.grad is None
    for parameter in actual.draft_model.mtp_fusion.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_nonzero_fusion_gradient_checkpointing_preserves_outputs_and_gradients(algorithm):
    actual = _nonzero_model(algorithm)
    reference = copy.deepcopy(actual)
    actual.draft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert actual.training and actual.draft_model.gradient_checkpointing
    result, expected = actual(**_fusion_inputs()), reference(**_fusion_inputs())
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    result[0].backward()
    expected[0].backward()
    _assert_parameter_gradients(actual, reference)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("all_inactive", [False, True])
def test_inactive_nan_mtp_features_preserve_finite_outputs_and_valid_gradients(algorithm, all_inactive):
    actual = _nonzero_model(algorithm)
    reference = copy.deepcopy(actual)
    inputs, reference_inputs = _fusion_inputs(), _fusion_inputs()
    for values in [inputs, reference_inputs]:
        values["injected_anchors"] = torch.tensor([[3, 0]])
        values["injected_keep_mask"] = torch.tensor([[not all_inactive, False]])
        values["attention_mask"] = torch.tensor([[1] * 8 + [0] * 4])
        values["loss_mask"] = values["attention_mask"].clone()
    with torch.no_grad():
        # Inactive anchor slots use row zero; padded feature rows are also irrelevant.
        inputs["draft_prefix"].prediction_hidden[:, 0] = float("nan")
        inputs["draft_prefix"].prediction_hidden[:, 7:] = float("nan")
    result, expected = actual(**inputs), reference(**reference_inputs)
    assert torch.isfinite(result[0])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result[0].backward()
    expected[0].backward()
    _assert_parameter_gradients(actual, reference, exact=True)


def test_residual_starts_at_token_only_and_only_changes_t1():
    model = wrapper()
    ids = torch.arange(12).unsqueeze(0)
    anchors, keep = torch.tensor([[3, 5]]), torch.tensor([[True, False]])
    base = model._create_noise_embed(ids, anchors, keep)
    hidden = torch.randn(1, 11, 16, requires_grad=True)
    actual = model._condition_noise_embed(base, ids, anchors, keep, MTPPrefix(hidden))
    expected = base.clone()
    expected[:, 1] = model.draft_model.embed_tokens(ids[:, 4])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[:, 1].sum().backward()
    assert model.draft_model.mtp_fusion.proj.weight.grad.abs().sum() > 0
    assert hidden.grad is None
    assert model.draft_model.embed_tokens.weight.grad is None


def test_fusion_selects_state_predicting_t1_and_accepts_actual_proposals():
    model = wrapper()
    ids = torch.arange(12).unsqueeze(0)
    anchors, keep = torch.tensor([[3, 5]]), torch.ones(1, 2, dtype=torch.bool)
    hidden = torch.arange(11 * 16).reshape(1, 11, 16).float()
    seen = []
    model.draft_model.mtp_fusion.register_forward_pre_hook(lambda _, values: seen.append(values))
    proposed = torch.tensor([[20, 21]])
    base = model._create_noise_embed(ids, anchors, keep)
    actual = model._condition_noise_embed(base, ids, anchors, keep, MTPPrefix(hidden, proposed))
    torch.testing.assert_close(seen[0][1], hidden[:, [2, 4]])
    torch.testing.assert_close(actual[:, [1, 6]], model.draft_model.embed_tokens(proposed))


def test_mtp_features_are_detached_even_after_residual_becomes_nonzero():
    fusion = MTPAnchorFusion(16, 16)
    nn.init.normal_(fusion.proj.weight)
    hidden = torch.randn(1, 16, requires_grad=True)
    fusion(torch.randn(1, 16), hidden).square().sum().backward()
    assert hidden.grad is None
    assert fusion.hidden_norm.weight.grad.abs().sum() > 0


def test_only_suffix_positions_receive_loss_and_decay_restarts_at_t2(monkeypatch):
    model = wrapper()
    features = torch.randn(1, 10, 16, requires_grad=True)
    anchors = torch.tensor([[3, 5]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    monkeypatch.setattr(model, "_draft_backbone", lambda *a, **k: (features, anchors, keep, 2))
    ids = torch.arange(12).unsqueeze(0)
    head = torch.randn(32, 16)
    result = model(input_ids=ids, hidden_states_list=None, loss_mask=torch.ones_like(ids),
                   lm_head_weight=head, draft_prefix=MTPPrefix(torch.randn(1, 11, 16)))
    logits = F.linear(features, head).reshape(1, 2, 5, 32)
    labels = ids[:, anchors[0, :, None] + torch.arange(5)]
    ce = F.cross_entropy(logits[:, :, 2:].flatten(0, 2), labels[:, :, 2:].flatten(),
                         reduction="none").reshape(1, 2, 3)
    weights = torch.exp(-torch.arange(3).float() / model.loss_decay_gamma)
    expected = (ce * weights).sum() / (2 * weights.sum())
    torch.testing.assert_close(result[0], expected)
    assert result[4].tolist() == [0, 0, 2, 2, 2]
    result[0].backward()
    assert features.grad.reshape(1, 2, 5, 16)[:, :, :2].count_nonzero() == 0


def test_context_cutoff_stays_before_a_even_with_t1_visible():
    mask = _create_dflash_mask_mod(torch.tensor([[3]]), torch.tensor([[True]]), 12, 5)
    for query in range(5):
        assert mask(0, 0, query, 2)
        assert not mask(0, 0, query, 3)
        assert not mask(0, 0, query, 4)
        assert mask(0, 0, query, 13)  # T1 is visible within the draft block.


def test_frozen_mtp_uses_previous_target_state_and_known_pending_token():
    class Predictor(nn.Module):
        def predict(self, hidden, ids, positions):
            self.seen = hidden, ids, positions
            return hidden + ids.unsqueeze(-1)
    mtp = Predictor().eval()
    ids = torch.arange(6).unsqueeze(0)
    final = torch.randn(1, 6, 16, requires_grad=True)
    output = frozen_mtp_predictions(mtp, ids, final)
    torch.testing.assert_close(mtp.seen[0], final[:, :-1])
    assert mtp.seen[1].tolist() == [[1, 2, 3, 4, 5]]
    assert mtp.seen[2].tolist() == [[0, 1, 2, 3, 4]]
    assert not output.requires_grad
    mtp.train()
    with pytest.raises(ValueError, match="frozen"):
        frozen_mtp_predictions(mtp, ids, final)


def test_invalid_timing_and_packed_input_fail_closed():
    model = wrapper()
    ids = torch.arange(12).unsqueeze(0)
    anchors, keep = torch.tensor([[0, 3]]), torch.ones(1, 2, dtype=torch.bool)
    base = model._create_noise_embed(ids, anchors, keep)
    with pytest.raises(ValueError, match="previous Target"):
        model._condition_noise_embed(base, ids, anchors, keep, MTPPrefix(torch.randn(1, 11, 16)))
    with pytest.raises(ValueError, match="packed"):
        model(input_ids=ids, hidden_states_list=None, loss_mask=torch.ones_like(ids),
              lm_head_weight=torch.randn(32, 16), ctx_doc_ids=torch.zeros_like(ids))
    with pytest.raises(ValueError, match="complete unpadded"):
        frozen_mtp_predictions(SimpleNamespace(training=False, parameters=lambda: []),
                               ids.expand(2, -1), torch.randn(2, 12, 16))


def test_frozen_provider_matches_native_and_materializes_meta_buffers(monkeypatch):
    from aspec.models.dmtp import FrozenMTPFeatures, training_mtp_prefix
    from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel

    expected = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    expected.requires_grad_(False).eval()
    tensors = {expected.native_key(k): v for k, v in expected.state_dict().items()}
    monkeypatch.setattr("aspec.models.draft.base.load_tensor_from_pretrained", lambda _, k: tensors[k])
    provider = FrozenMTPFeatures(native_config())
    provider.initialize_native_weights("unused")
    provider.to_empty(device="cpu")
    provider.initialize_native_weights("unused")
    provider.train()
    assert not provider.training and not any(p.requires_grad for p in provider.parameters())
    ids, final = torch.randint(0, 32, (1, 7)), torch.randn(1, 7, 16)
    actual = provider(final[:, :-1], expected.embed_tokens(ids[:, 1:]), torch.arange(6)[None])
    torch.testing.assert_close(actual, frozen_mtp_predictions(expected, ids, final), rtol=0, atol=0)
    provider.validate_native_weights("unused")
    draft = wrapper().draft_model
    draft.mtp_source = provider
    draft.embed_tokens.load_state_dict(expected.embed_tokens.state_dict())
    prefix = training_mtp_prefix(draft, ids, final)
    torch.testing.assert_close(prefix.prediction_hidden, actual, rtol=0, atol=0)
    shorter = training_mtp_prefix(draft, ids, final, torch.tensor([[1, 1, 1, 1, 0, 0, 0]]))
    torch.testing.assert_close(shorter.prediction_hidden[:, :3], actual[:, :3])
    assert shorter.prediction_hidden[:, 3:].count_nonzero() == 0
    assert not any(k.startswith(("embed_tokens", "lm_head")) for k in provider.state_dict())


@pytest.mark.parametrize("algorithm,start", [("dflash", 2), ("dspark", 1)])
def test_native_policy_is_checkpoint_identity_and_suffix_supervision(algorithm, start, monkeypatch):
    from aspec.config.draft_algorithm import draft_model_semantics
    from aspec.models.draft.auto import AutoDraftModelConfig

    model = wrapper(algorithm=algorithm)
    cfg = model.draft_model.config
    semantics = draft_model_semantics(cfg)
    assert semantics["mtp_conditioning"] == cfg.mtp_conditioning
    assert draft_model_semantics(semantics) == semantics
    restored = AutoDraftModelConfig.from_dict(cfg.to_dict())
    assert restored.mtp_conditioning == cfg.mtp_conditioning
    features = torch.randn(1, 10, 16, requires_grad=True)
    anchors, keep = torch.tensor([[3, 5]]), torch.ones(1, 2, dtype=torch.bool)
    monkeypatch.setattr(model, "_draft_backbone", lambda *a, **k: (features, anchors, keep, 2))
    ids = torch.arange(12)[None]
    output = model(ids, None, torch.ones_like(ids), torch.randn(32, 16),
                   last_hidden_states=torch.randn(1, 12, 16),
                   draft_prefix=MTPPrefix(torch.randn(1, 11, 16)))
    assert model.prediction_start == start
    assert output[4][:start].count_nonzero() == 0
    assert output[4][start:].tolist() == [2] * (5 - start)
    output[0].backward()
    assert features.grad.view(1, 2, 5, 16)[:, :, :start].count_nonzero() == 0
    if algorithm == "dspark":
        assert "tau_probabilistic" not in output[5]
        assert "suffix_tau_probabilistic" in output[5]
        assert len([k for k in output[5] if k.startswith("suffix_accept_rate")]) == 4


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("mode", [None, "token", "hidden"])
@pytest.mark.parametrize("namespace", ["train", "eval"])
def test_prediction_metrics_identify_suffix_scope(algorithm, mode, namespace, monkeypatch):
    from aspec.training.dflash_trainer import DFlashTrainer
    from aspec.training.dspark_trainer import DSparkTrainer

    cls = DFlashTrainer if algorithm == "dflash" else DSparkTrainer
    trainer = object.__new__(cls)
    conditioned = mode is not None
    trainer.dflash = SimpleNamespace(mtp_conditioning={"mode": mode} if conditioned else None)
    trainer._anchor_slot_offset = int(algorithm == "dflash") + int(conditioned)
    trainer._extra_loss_component_keys = []
    trainer.loss_decay_gamma, trainer.global_step = 1.0, 7
    trainer.optimizer = SimpleNamespace(get_learning_rate=lambda: 0.001)
    monkeypatch.setattr("aspec.training.dflash_trainer.dist.get_rank", lambda: 0)
    monkeypatch.setattr("aspec.training.dflash_trainer.dist.all_reduce", lambda *a, **k: None)
    offset = trainer._anchor_slot_offset
    losses = torch.tensor([0.0] * offset + [1.0, 2.0, 3.0])
    accuracies = torch.tensor([0.0] * offset + [0.5, 0.5, 0.5])
    counts = torch.tensor([0.0] * offset + [1.0, 1.0, 1.0])
    if namespace == "train":
        result = trainer._aggregate_metrics([dict(loss_per_position=losses, acc_per_position=accuracies,
                                                  count_per_position=counts)], 7, grad_norm=torch.tensor(0.0))
    else:
        result = trainer._aggregate_eval_metrics([dict(loss_pp=losses, acc_pp=accuracies, count_pp=counts)])
    prefix = "suffix_" if conditioned else ""
    assert result[f"{namespace}/{prefix}simulated_acc_len"] == pytest.approx(0.875)
    assert result[f"{namespace}/{prefix}ploss_0"] == 1.0
    assert result[f"{namespace}/{prefix}acc_0"] == 0.5
    opposite = "" if conditioned else "suffix_"
    for name in ["simulated_acc_len", "ploss_0", "acc_0"]:
        assert f"{namespace}/{opposite}{name}" not in result


@pytest.mark.parametrize("metric", ["eval/simulated_acc_len", "eval/suffix_simulated_acc_len"])
def test_checkpoint_selection_persists_the_observed_metric_scope(tmp_path, metric):
    from aspec.controller.eval import eval_selection_metric, update_checkpoint_eval_meta

    checkpoint = tmp_path / "iter_0000002"
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(json.dumps({"global_step": 1, "eval/simulated_acc_len": 0.1}))
    metrics = {"eval/avg_loss": 1.0, "eval/avg_acc": 0.5, metric: 0.875}
    assert eval_selection_metric(metrics) == metric
    assert update_checkpoint_eval_meta(str(tmp_path), 1, metrics, -float("inf")) == 0.875
    assert (tmp_path / "best_checkpointed_iteration.txt").read_text() == "2"
    for path in [checkpoint / "meta.json", tmp_path / "best_meta.json"]:
        metadata = json.loads(path.read_text())
        assert metadata[metric] == 0.875
        assert metadata["selection_metric"] == metric
        if metric == "eval/suffix_simulated_acc_len":
            assert "eval/simulated_acc_len" not in metadata


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_strict_native_asset_and_weights_only_transition(tmp_path, monkeypatch, algorithm):
    from safetensors.torch import save_file

    from aspec.commands import convert_to_hf
    from aspec.draft_artifact import BlockDraftConsumer, load_native_draft
    from aspec.training.checkpoint import load_initial_draft_weights

    model = _nonzero_model(algorithm).eval()
    draft = model.draft_model
    # A real source initialized from the native predictor, without a Target head copy.
    from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel

    native = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    state = {native.native_key(k): v for k, v in native.state_dict().items()}
    monkeypatch.setattr("aspec.models.draft.base.load_tensor_from_pretrained", lambda _, k: state[k])
    draft.initialize_mtp_conditioning("unused")
    inputs = _fusion_inputs()
    inputs.pop("draft_prefix")
    backbone_outputs = []
    handle = draft.register_forward_hook(lambda _, args, output: backbone_outputs.append(output))
    with torch.no_grad():
        before = model(**inputs)
    handle.remove()
    root = tmp_path / "iter_0000003"
    (root / "model").mkdir(parents=True)
    (root / "model/.metadata").touch()
    (root / "draft_config.json").write_text(json.dumps(draft.config.to_dict()))
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: {
        "draft_model." + key: value for key, value in draft.state_dict().items()})
    output = tmp_path / "native"
    manifest = convert_to_hf.convert_checkpoint(input_dir=root, output_dir=output, artifact_format="aspec")
    assert manifest["draft_structure"]["loader_contract"]["proposal_entrypoint"] is None
    restored = load_native_draft(output)
    assert not restored.embed_tokens.weight.requires_grad
    assert not any(p.requires_grad for p in restored.mtp_source.parameters())
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    restored_model = type(model)(restored, block_size=5, num_anchors=2, attention_backend="sdpa").eval()
    handle = restored.register_forward_hook(lambda _, args, output: backbone_outputs.append(output))
    with torch.no_grad():
        torch.testing.assert_close(restored_model(**inputs), before, rtol=0, atol=0)
    handle.remove()
    torch.testing.assert_close(backbone_outputs[1], backbone_outputs[0], rtol=0, atol=0)
    logits, _ = _reference_logits_and_loss(model, backbone_outputs[0], inputs)
    restored_logits, _ = _reference_logits_and_loss(restored_model, backbone_outputs[1], inputs)
    torch.testing.assert_close(restored_logits, logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="MTP-aware"):
        BlockDraftConsumer(restored, block_size=5)
    with pytest.raises(NotImplementedError, match="native"):
        convert_to_hf._serving_config(draft.config.to_dict())
    plain = tmp_path / "plain"
    plain.mkdir()
    cfg = {**draft.config.to_dict(), "mtp_conditioning": None}
    (plain / "config.json").write_text(json.dumps(cfg))
    tensors = {k: v for k, v in draft.state_dict().items()
               if not k.startswith(("mtp_source.", "mtp_fusion."))}
    save_file(tensors, str(plain / "model.safetensors"))
    load_initial_draft_weights(draft, str(plain))
    tensors["mtp_fusion.proj.weight"] = draft.mtp_fusion.proj.weight
    save_file(tensors, str(plain / "model.safetensors"))
    with pytest.raises(ValueError, match="undeclared"):
        load_initial_draft_weights(draft, str(plain))


@pytest.mark.parametrize("key,value", [
    ("micro_batch_size", 2), ("draft_context_parallel_size", 2),
    ("dflash_packing", True), ("compile_model", True), ("fsdp_strategy", "FULL_SHARD"),
])
def test_conditioned_plan_rejects_unimplemented_combinations(key, value, monkeypatch):
    from pathlib import Path

    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    monkeypatch.setenv("ASPEC_TRAIN_DATA", "examples/data/sample_conversations.jsonl")
    monkeypatch.setenv("ASPEC_EVAL_DATA", "examples/data/sample_conversations.jsonl")
    path = Path(__file__).parents[2] / "configs/recipes/qwen3_5_9b_dmtp.yaml"
    config = load_config(str(path), save_snapshot=False)
    plan = compile_execution_plan(config)
    assert plan.workload.objective["prediction_start"] == 1
    assert plan.workload.objective["suffix_prediction_budget"] == 6
    config.training[key] = value
    with pytest.raises(ValueError):
        compile_execution_plan(config)


def test_native_condition_never_reads_future_tokens_or_target_states(monkeypatch):
    from aspec.models.dmtp import FrozenMTPFeatures
    from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel

    native = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config()))
    tensors = {native.native_key(k): v for k, v in native.state_dict().items()}
    monkeypatch.setattr("aspec.models.draft.base.load_tensor_from_pretrained", lambda _, k: tensors[k])
    provider = FrozenMTPFeatures(native_config())
    provider.initialize_native_weights("unused")
    ids = torch.randint(0, 32, (1, 8))
    hidden = torch.randn(1, 8, 16)
    positions = torch.arange(8)[None]
    before = provider(hidden, native.embed_tokens(ids), positions)
    changed_ids, changed_hidden = ids.clone(), hidden.clone()
    changed_ids[:, 4:] = (changed_ids[:, 4:] + 1) % 32
    changed_hidden[:, 4:] += 10
    after = provider(changed_hidden, native.embed_tokens(changed_ids), positions)
    torch.testing.assert_close(before[:, :4], after[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_zero_hidden_residual_matches_token_conditioned_workload(algorithm):
    token = wrapper(use_hidden=False, algorithm=algorithm)
    hidden = wrapper(use_hidden=True, algorithm=algorithm)
    with torch.no_grad():
        for name, parameter in token.draft_model.named_parameters():
            if not parameter.is_meta:
                hidden.draft_model.get_parameter(name).copy_(parameter)
    ids = torch.arange(12)[None]
    kwargs = dict(input_ids=ids, hidden_states_list=torch.randn(1, 12, 16),
                  loss_mask=torch.ones_like(ids), lm_head_weight=torch.randn(32, 16),
                  last_hidden_states=torch.randn(1, 12, 16),
                  injected_anchors=torch.tensor([[3, 5]]),
                  injected_keep_mask=torch.ones(1, 2, dtype=torch.bool))
    expected = token(**kwargs, draft_prefix=MTPPrefix())
    actual = hidden(**kwargs, draft_prefix=MTPPrefix(torch.randn(1, 11, 16)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[0].backward()
    assert hidden.draft_model.mtp_fusion.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in hidden.draft_model.mtp_source.parameters())
