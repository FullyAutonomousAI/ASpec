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

"""Pruned head mathematics, immutable mappings and strict export boundaries."""

import copy
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file

from aspec.config.train_config import load_config
from aspec.models import eagle3 as eagle
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.vocab import validate_vocab_mapping
from aspec.planning import compile_execution_plan
from tools import convert_to_hf

ROOT = Path(__file__).parents[1]


def raw_config():
    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    raw.update(hidden_size=16, target_hidden_size=16, intermediate_size=32,
               num_attention_heads=2, num_key_value_heads=1, head_dim=8,
               vocab_size=32, draft_vocab_size=17, pad_token_id=0)
    return raw


def mapping():
    selected = torch.tensor(list(range(0, 32, 2)) + [31])
    membership = torch.zeros(32, dtype=torch.bool)
    membership[selected] = True
    return selected - torch.arange(17), membership


@pytest.fixture
def eager_losses(monkeypatch):
    # Unit tests cover mathematics, not compiler startup latency.
    for name in dir(eagle):
        fn = getattr(eagle, name)
        if name.startswith("compiled_"):
            monkeypatch.setattr(eagle, name, getattr(fn, "_torchdynamo_orig_callable", fn))


@pytest.mark.parametrize("kind", ["forward_kl", "lk_alpha", "lk_lambda"])
def test_lazy_pruning_matches_retained_probability_reference_and_gradients(kind, eager_losses):
    torch.manual_seed(19)
    draft = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw_config()))
    d2t, t2d = mapping()
    draft.set_vocab_buffers(d2t, t2d)
    model = eagle.Eagle3Model(draft, length=3, loss_type=kind)
    teacher = torch.randn(2, 271, 16)
    target_head = torch.randn(32, 16)
    mask = torch.randint(0, 2, (2, 271)).float()
    reference = eagle.compute_target_p_padded(teacher, target_head, t2d, mask, 3, chunk_size=128)
    lazy = eagle.compute_lazy_target_padded(teacher, target_head, 3, t2d)
    lazy.pruned_weight = target_head[t2d]
    norm, head, eps = draft.get_lm_head_params()
    states = torch.randn(2, 271, 16, requires_grad=True)
    expected_mask = torch.nn.functional.pad(reference.position_mask[:, 1:], (0, 1))
    actual_mask = torch.nn.functional.pad(mask[:, 1:], (0, 1))
    expected = model._calculate_loss(states, reference, expected_mask, 1, 271, norm, head, eps, None)
    expected_grad = torch.autograd.grad(expected[0], (states, norm, head))
    actual = model._calculate_loss(states, lazy, actual_mask, 1, 271, norm, head, eps, None)
    actual_grad = torch.autograd.grad(actual[0], (states, norm, head))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-6)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


def test_empty_pruned_owner_participates_once_in_depth_lambda_collective(eager_losses, monkeypatch):
    draft = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw_config()))
    model = eagle.Eagle3Model(draft, length=3, loss_type="lk_lambda")
    model._usp_sp_group = object()
    reduce = Mock()
    monkeypatch.setattr(eagle.dist, "all_reduce", reduce)
    t2d = mapping()[1]
    target = eagle.LazyTarget(torch.randn(1, 3, 16), torch.randn(32, 16), t2d)
    target.pruned_weight = target.lm_head_weight[t2d]
    norm, head, eps = draft.get_lm_head_params()
    result = model._calculate_loss(torch.randn(1, 3, 16), target, torch.zeros(1, 3), 0, 3,
                                   norm, head, eps, None)
    result[0].backward()
    assert reduce.call_count == 1
    assert result[2] == 0
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("damage", ["dtype", "count", "rows", "shape"])
def test_mapping_validation_rejects_unbound_or_corrupt_head(damage):
    d2t, t2d = mapping()
    if damage == "dtype":
        t2d = t2d.long()
    elif damage == "count":
        t2d[1] = True
    elif damage == "rows":
        d2t = d2t.flip(0)
    else:
        d2t = d2t[:-1]
    with pytest.raises(ValueError, match="Vocabulary mapping"):
        validate_vocab_mapping(d2t, t2d, 17, 32)


def test_published_pruned_head_keeps_mapping_and_rejects_reassignment(tmp_path, monkeypatch):
    raw = raw_config()
    source = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    source.set_vocab_buffers(*mapping())
    model_dir = tmp_path / "iter_0000017" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / ".metadata").touch()
    config = model_dir.parent / "draft_config.json"
    config.write_text(json.dumps(raw))
    weights = {"draft_model." + k: v for k, v in source.state_dict().items()}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: weights)
    exported = tmp_path / "exported"
    manifest = convert_to_hf.convert_checkpoint(input_dir=model_dir.parent, output_dir=exported)
    tensors = load_file(exported / "model.safetensors")
    assert {"d2t", "t2d"} <= tensors.keys()
    assert manifest["vocabulary"]["draft_vocab_size"] == 17
    from aspec.training.checkpoint import load_initial_draft_weights
    restored = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    load_initial_draft_weights(restored, str(exported))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    d2t, t2d = mapping()
    t2d[0], t2d[1] = False, True
    d2t = t2d.nonzero().flatten() - torch.arange(17)
    with pytest.raises(ValueError, match="different token set"):
        restored.set_vocab_buffers(d2t, t2d)


def test_export_cannot_invent_missing_mapping(tmp_path, monkeypatch):
    raw = raw_config()
    model = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / ".metadata").touch()
    (tmp_path / "draft_config.json").write_text(json.dumps(raw))
    weights = {"draft_model." + k: v for k, v in model.state_dict().items() if k not in {"d2t", "t2d"}}
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: weights)
    with pytest.raises(ValueError, match="mapping"):
        convert_to_hf.convert_checkpoint(input_dir=tmp_path, output_dir=tmp_path.parent / (tmp_path.name + "-out"))


def test_loaded_heads_do_not_recompute_mapping(tmp_path, monkeypatch):
    from aspec.train_entry import _prepare_vocab_mapping
    from aspec.training import checkpoint

    controller = Mock()
    config = Namespace(vocab_size=32, draft_vocab_size=17)
    monkeypatch.setattr(checkpoint, "resolve_resume_model_dir", lambda _: tmp_path)
    assert _prepare_vocab_mapping(Namespace(), config, controller) is None
    monkeypatch.setattr(checkpoint, "resolve_resume_model_dir", lambda _: None)
    assert _prepare_vocab_mapping(Namespace(initial_draft_model_path="draft"), config, controller) is None
    controller.compute_vocab_mapping.remote.assert_not_called()


def test_public_pruned_eagle_plan_accepts_packing_cp_without_changing_default(tmp_path):
    raw = json.loads((ROOT / "aspec/config/eagle3_qwen3_8b_draft_config.json").read_text())
    raw["draft_vocab_size"] = 32000
    path = tmp_path / "pruned.json"
    path.write_text(json.dumps(raw))
    config = load_config(str(ROOT / "configs/recipes/qwen3_8b_eagle3.yaml"), save_snapshot=False)
    old = copy.deepcopy(config)
    config.model.draft_model_config = str(path)
    config.training.attention_backend = "fa4"
    config.training.eagle3_target_cache = "lazy"
    config.training.eagle3_loss_gradient_checkpointing = True
    config.training.dflash_packing = True
    config.training.draft_context_parallel_size = 2
    plan = compile_execution_plan(config)
    assert plan.workload.objective["family"] == "eagle3"
    assert old.model.draft_model_config != str(path)


def test_published_pruned_mapping_cannot_be_corrupt_even_with_matching_shapes(tmp_path):
    from safetensors.torch import save_file

    from aspec.training.checkpoint import load_initial_draft_weights

    cfg = AutoDraftModelConfig.from_dict(raw_config())
    model = AutoEagle3DraftModel.from_config(cfg)
    model.set_vocab_buffers(*mapping())
    state = model.state_dict()
    state["d2t"] = state["d2t"].flip(0)
    asset = tmp_path / "asset"
    asset.mkdir()
    save_file(state, str(asset / "model.safetensors"))
    with pytest.raises(ValueError, match="Vocabulary mapping"):
        load_initial_draft_weights(AutoEagle3DraftModel.from_config(cfg), str(asset))


def test_pruned_teacher_head_is_cached_only_after_mapping_is_bound():
    from types import SimpleNamespace

    from aspec.training.eagle3_trainer import Eagle3Trainer

    draft = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw_config()))
    trainer = object.__new__(Eagle3Trainer)
    trainer.eagle3 = SimpleNamespace(draft_model=draft)
    trainer.t2d = draft.t2d
    trainer.target_lm_head_weight = torch.randn(32, 16)
    with pytest.raises(RuntimeError, match="initialized vocabulary mapping"):
        trainer._pruned_target_weight()
    draft.set_vocab_buffers(*mapping())
    first = trainer._pruned_target_weight()
    second = trainer._pruned_target_weight()
    assert first is second
    assert not first.requires_grad
    torch.testing.assert_close(first, trainer.target_lm_head_weight[trainer.t2d], rtol=0, atol=0)


def test_lazy_target_reuses_an_explicitly_bound_pruned_teacher_head(monkeypatch):
    draft = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw_config()))
    draft.set_vocab_buffers(*mapping())
    wrapper = eagle.Eagle3Model(draft, length=3)
    weight = torch.randn(32, 16)
    pruned = weight[draft.t2d]
    target = eagle.LazyTarget(torch.randn(1, 4, 16), weight, draft.t2d, pruned)
    monkeypatch.setattr(wrapper, "_forward_thd", lambda _hs, teacher, _layout: teacher.pruned_weight)
    actual = wrapper(torch.ones(1, 4, dtype=torch.long), None, target, torch.ones(1, 4),
                     torch.randn(1, 4, 48), thd_layout=object())
    assert actual is pruned
