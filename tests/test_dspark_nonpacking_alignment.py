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

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from tools.validation.dspark_nonpacking_alignment import (
    additive_terms,
    compare,
    normalized_terms,
    own_nonpacked,
    position_totals,
    preserve_cutlass_compilation,
    prompt_identity,
    reference_resume_location,
    resumed_optimizer_audit,
    run,
    split_heldout,
    training_acceptance,
    vector_comparison,
)


def test_comparison_rejects_old_runtime_patch_before_loading_fixture_or_gpus(monkeypatch):
    from aspec.compat import sglang_patch as apply_sglang_patch

    monkeypatch.setattr(apply_sglang_patch, "installed_site", lambda: "isolated-site")
    monkeypatch.setattr(apply_sglang_patch, "patch_state", lambda site: "unpatched")
    with pytest.raises(ValueError, match="current ASpec SGLang training patch"):
        run(SimpleNamespace(fixture="missing-fixture"))


def test_reference_import_restores_host_compiler_hooks_even_on_failure(monkeypatch):
    cached, compiled = object(), object()
    base = SimpleNamespace(compile_and_cache=cached)
    compiler = SimpleNamespace(_compile=compiled)
    for name, attribute, value in (
        ("cutlass.base_dsl.dsl", "BaseDSL", base),
        ("cutlass.base_dsl.compiler", "CompileCallable", compiler),
    ):
        module = types.ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="import failed"):
        with preserve_cutlass_compilation():
            base.compile_and_cache = object()
            compiler._compile = object()
            raise RuntimeError("import failed")
    assert base.compile_and_cache is cached
    assert compiler._compile is compiled


def row(prompt, suffix=99):
    return {
        "input_ids": [*prompt, suffix],
        "source_metadata_json": json.dumps({"prompt_input_ids": prompt}),
    }


def test_split_is_order_independent_and_never_overlaps_training():
    train = [row([1, 2])]
    heldout = [row([3]), row([4]), row([5])]
    val, test = split_heldout(train, heldout, count=1)
    assert (val, test) == split_heldout(train, heldout[::-1], count=1)
    assert len(val) == 2 and len(test) == 1
    with pytest.raises(ValueError, match="overlapping"):
        split_heldout(train, heldout + train, count=1)
    with pytest.raises(ValueError, match="Duplicate"):
        split_heldout(train, heldout + [heldout[0]], count=1)
    with pytest.raises(ValueError, match="prefix"):
        prompt_identity({"input_ids": [2], "source_metadata_json": '{"prompt_input_ids":[1]}'})


def test_nonpacked_replay_preserves_padding_and_owns_receive_buffers():
    ids = torch.arange(8).view(1, -1)
    features = torch.arange(48).reshape(1, 8, 6).float()
    values = dict(
        input_ids=ids,
        loss_mask=ids.float(),
        hidden_states_list=features,
        last_hidden_states=features[:, :, :2],
        injected_anchors=torch.tensor([[1, 2]]),
        injected_keep_mask=torch.tensor([[True, False]]),
    )
    actual = own_nonpacked(values, 2)
    assert len(actual["hidden_states_list"]) == 3
    assert torch.equal(torch.cat(actual["hidden_states_list"], -1), features)
    ids.zero_()
    features.zero_()
    assert actual["input_ids"][0, -1] == 7
    assert actual["hidden_states_list"][-1][0, -1, -1] == 47
    assert actual["last_hidden_states"][0, -1, -1] == 43
    with pytest.raises(ValueError, match="non-packed"):
        own_nonpacked({**values, "ctx_doc_ids": torch.zeros_like(ids)}, 2)


def test_full_vector_comparison_does_not_confuse_equal_norms_with_alignment():
    a = [torch.tensor([1.0, -1.0])]
    b = [torch.tensor([-1.0, 1.0])]
    result = vector_comparison(iter(a), iter(b))
    assert result["actual_norm"] == result["reference_norm"]
    assert result["relative_l2"] == 2
    assert result["max_absolute"] == 2
    with pytest.raises(ValueError, match="presence"):
        vector_comparison([None], b)


def test_additive_objective_uses_pooled_denominator_not_document_means():
    totals = []
    for denominator, ce in ((1.0, 2.0), (3.0, 9.0)):
        components = {
            "ce_loss": (torch.tensor(ce), denominator),
            "l1_loss": (torch.tensor(0.0), denominator),
            "confidence_loss": (torch.tensor(0.0), denominator),
        }
        output = (None,) * 5 + (components, (torch.tensor(0.1 * ce), torch.tensor(denominator)))
        totals.append(additive_terms(output))
    result = normalized_terms(torch.stack(totals).sum(0).tolist())
    assert result["ce"] == 11 / 4
    assert result["ce"] != (2 + 3) / 2
    assert result["denominator"] == 4
    with pytest.raises(ValueError, match="empty"):
        normalized_terms([0.0] * 5)
    with pytest.raises(ValueError, match="reconstruct"):
        normalized_terms([1.0, 1.0, 0.0, 0.0, 0.0])


def test_training_acceptance_uses_pooled_counts_and_is_not_decode_acceptance():
    totals = []
    for count, accuracy in ((1.0, 1.0), (3.0, 0.0)):
        output = (
            None,
            None,
            torch.tensor([2.0, 3.0]),
            torch.tensor([accuracy, accuracy]),
            torch.tensor([count, count]),
        )
        totals.append(position_totals(output))
    actual = training_acceptance(torch.stack(totals).sum(0), {"l1": 1.2})
    assert actual["per_position_accuracy"] == [0.25, 0.25]
    assert actual["per_position_count"] == [4.0, 4.0]
    assert actual["token_accuracy"] == 0.25
    assert actual["simulated_accepted_draft_tokens"] == 0.25 + 0.25**2
    assert actual["weighted_distribution_overlap"] == 0.4
    assert "not decode acceptance" in actual["convention"]
    with pytest.raises(ValueError, match="supervision"):
        training_acceptance(torch.zeros(3, 7), {"l1": 1.0})


def test_resume_audit_requires_real_moments_phase_and_rounded_masters():
    model = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    master = torch.nn.Parameter(model.float().detach())
    slot = {"step": torch.tensor(32.0), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}
    scheduler = SimpleNamespace(last_epoch=32, state_dict=lambda: {"last_epoch": 32})
    optimizer = SimpleNamespace(
        state={master: slot},
        model_params=[model],
        fp32_params=[master],
        lr_scheduler=scheduler,
        get_learning_rate=lambda: 0.0004,
    )
    audit = resumed_optimizer_audit(optimizer, 32)
    assert audit["adam_slots"] == 1 and audit["adam_steps"] == [32]
    assert len(audit["masters_sha256"]) == 1
    assert set(audit["moments_sha256"][0]) == {"exp_avg", "exp_avg_sq"}
    slot["step"] = torch.tensor(31.0)
    with pytest.raises(ValueError, match="counter"):
        resumed_optimizer_audit(optimizer, 32)
    slot["step"] = torch.tensor(32.0)
    scheduler.last_epoch = 31
    with pytest.raises(ValueError, match="LR phase"):
        resumed_optimizer_audit(optimizer, 32)
    scheduler.last_epoch = 32
    model.data.add_(1)
    with pytest.raises(ValueError, match="masters"):
        resumed_optimizer_audit(optimizer, 32)


def test_native_reference_explicit_frontier_uses_parent_and_step(tmp_path):
    checkpoint = tmp_path / "iter_0000032"
    with pytest.raises(ValueError, match="missing"):
        reference_resume_location(checkpoint)
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").touch()
    assert reference_resume_location(checkpoint) == (str(tmp_path), 32)
    assert reference_resume_location(tmp_path) == (str(tmp_path), None)


def test_comparison_requires_all_ranks_and_exact_supervision(tmp_path):
    terms = {"loss": 1.0, "ce": 1.0, "l1": 0.0, "confidence": 0.9, "denominator": 12.0}
    reference = {**terms, "gradients": {"relative_l2": 0.01}}
    rows = [
        {
            "step": index + 1,
            "aspec": terms,
            "common_weight": reference,
            "torchspec": reference,
            "weights": {"relative_l2": 0.001},
            "updates": {"relative_l2": 0.02},
        }
        for index in range(32)
    ]
    for rank in range(4):
        (tmp_path / f"steps-rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    compare(SimpleNamespace(run_dir=tmp_path, steps=32))
    result = json.loads((tmp_path / "comparison.json").read_text())["summary"]
    assert result["documents"] == 16384
    assert result["metrics"]["common_weight"]["loss"]["max_relative_diff"] == 0
    assert result["updates_max_relative_l2"] == 0.02
    rows[-1] = {**rows[-1], "common_weight": {**reference, "denominator": 13.0}}
    (tmp_path / "steps-rank0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="denominator"):
        compare(SimpleNamespace(run_dir=tmp_path, steps=32))
    (tmp_path / "steps-rank3.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1])
    )
    with pytest.raises(ValueError, match="Missing"):
        compare(SimpleNamespace(run_dir=tmp_path, steps=32))
