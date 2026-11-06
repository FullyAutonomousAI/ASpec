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

import pytest
import torch

from aspec.config.train_config import load_config
from aspec.models.mtp import MTPModel
from aspec.models.ops.eagle_thd import EagleTHDLayout
from aspec.planning import compile_execution_plan
from tests.test_native_mtp import ROOT, model


def test_native_mtp_fa4_packing_cp_is_a_public_plan():
    from aspec.config.draft_attention import resolve_draft_attention

    config = load_config(str(ROOT / "configs/recipes/qwen3_5_9b_mtp.yaml"), save_snapshot=False)
    config.training.attention_backend = "fa4"
    config.training.dflash_packing = True
    config.training.draft_context_parallel_size = 2
    config.training.training_num_gpus_per_node = 2
    plan = compile_execution_plan(config)
    assert plan.to_runtime_args().draft_context_parallel_size == 2
    assert plan.workload.objective["family"] == "mtp"
    observed = resolve_draft_attention(model().config.to_dict())
    assert observed["draft_block_attention"] == "causal_native_mtp"
    assert plan.workload.models["draft"]["attention"]["draft_block_attention"] == observed["draft_block_attention"]


def test_legacy_native_mtp_policy_label_is_normalized_only_at_resume_boundary():
    import copy

    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    config = load_config(str(ROOT / "configs/recipes/qwen3_5_9b_mtp.yaml"), save_snapshot=False)
    config.training.training_num_gpus_per_node = 1
    args = compile_execution_plan(config).to_runtime_args()
    args.draft_model_config_obj = model().config
    saved = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    saved["training_topology"]["draft_attention_policy"]["draft_block_attention"] = "bidirectional"
    original = copy.deepcopy(saved)
    _validate_exact_resume_topology(args, saved)
    assert saved == original
    saved["training_topology"]["draft_attention_policy"]["window_semantics"] = "query_relative"
    with pytest.raises(RuntimeError, match="logical training-topology changes"):
        _validate_exact_resume_topology(args, saved)
    saved = copy.deepcopy(original)
    saved["training_topology"]["draft_model_semantics"]["is_causal"] = False
    with pytest.raises((RuntimeError, ValueError), match="causal|training-topology"):
        _validate_exact_resume_topology(args, saved)


@pytest.mark.parametrize("lengths", [[13, 9], [2, 3, 5, 11]])
def test_packed_mtp_keeps_native_prefix_labels_and_complete_gradients(lengths):
    """A document SDPA stand-in checks layout/math without loading GPU kernels."""
    from types import MethodType

    torch.manual_seed(44)
    draft, reference = model(), model()
    reference.load_state_dict(draft.state_dict())
    ids = torch.randint(0, 32, (1, sum(lengths)))
    hidden = torch.randn(1, sum(lengths), 16)
    mask = torch.randint(0, 2, ids.shape)
    positions = torch.arange(sum(lengths))
    layout = EagleTHDLayout(lengths, ids.flatten(), mask.flatten(), positions)
    native_predict = draft.predict

    def predict(self, states, tokens, rope, layout):
        rows, start = [], 0
        for length in lengths:
            steps = min(3, max(0, length - 2))
            size = length - steps - 1 if steps else 0
            value = native_predict(states[:, start:start + size], tokens[:, start:start + size],
                                   rope[:, start:start + size]) if size else states[:, :0]
            rows.append(torch.nn.functional.pad(value, (0, 0, 0, length - size)))
            start += length
        return torch.cat(rows, dim=1)

    draft.predict = MethodType(predict, draft)
    actual = MTPModel(draft, length=3)(None, hidden, None, None, thd_layout=layout)
    rows, start = [], 0
    for length in lengths:
        rows.append(MTPModel(reference, length=3)(
            ids[:, start:start + length], hidden[:, start:start + length],
            mask[:, start:start + length], torch.ones(1, length),
        ))
        start += length
    for depth in range(3):
        count = sum(row[3][depth] for row in rows)
        expected = sum(row[0][depth] * row[3][depth] for row in rows) / count.clamp_min(1)
        torch.testing.assert_close(actual[0][depth], expected)
        if depth == 0:
            expected_losses = []
        expected_losses.append(expected)
    sum(actual[0]).backward()
    sum(expected_losses).backward()
    for (name, a), (_, b) in zip(draft.named_parameters(), reference.named_parameters(), strict=True):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, msg=name)
