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

import pytest
import torch

from tools.validation.dspark_torchspec_alignment import (
    TorchReference,
    compare,
    document_batches,
    loss_record,
)


def test_per_document_replay_preserves_identity_and_owns_its_copies():
    ids = torch.arange(13).view(1, -1)
    kwargs = {
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids, dtype=torch.float32),
        "ctx_doc_ids": torch.tensor([[0] * 5 + [1] * 7 + [-1]]),
        "hidden_states_list": [ids.unsqueeze(-1).float()],
        "last_hidden_states": ids.unsqueeze(-1).float(),
        "injected_anchors": torch.tensor([[0, 2, 5, 8, 0]]),
        "injected_keep_mask": torch.tensor([[True, True, True, True, False]]),
    }
    first, second = document_batches(kwargs)
    assert first["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert second["input_ids"].tolist() == [[5, 6, 7, 8, 9, 10, 11]]
    assert first["anchors"].tolist() == [[0, 2, 0, 0, 0]]
    assert second["anchors"].tolist() == [[0, 0, 0, 3, 0]]
    assert first["keep"].sum() == second["keep"].sum() == 2
    ids.zero_()
    assert second["input_ids"][0, 0] == 5


def test_loss_pools_additive_terms_not_document_means():
    actual = loss_record([[2.0, 1.0, 1.0, 1.0, 1.0], [9.0, 3.0, 3.0, 3.0, 3.0]])
    assert actual["loss"] == 11 / 4
    assert actual["loss"] != (2 + 3) / 2
    with pytest.raises(ValueError, match="Empty"):
        loss_record([[0, 0, 0, 0, 0]])


def test_pre_forward_collects_without_nested_backward(monkeypatch):
    reference = TorchReference.__new__(TorchReference)
    reference.documents = []
    row = {"owned": "document"}
    monkeypatch.setattr(
        "tools.validation.dspark_torchspec_alignment.document_batches", lambda _: [row]
    )
    reference.begin = lambda: pytest.fail("Cannot run reference backward in an FSDP pre-hook")
    reference.compute_document = reference.begin
    reference.pre_forward(None, (), {"lm_head_weight": torch.ones(2, requires_grad=True)})
    assert reference.documents == [row]
    assert not reference.lm_head.requires_grad


def test_document_control_uses_native_accumulation_and_tensor_denominator():
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))
            self.sync = []

        def set_requires_gradient_sync(self, enabled):
            self.sync.append(enabled)

        def forward(self, **kwargs):
            numerator = self.parameter.sum()
            denominator = torch.tensor(1.25, dtype=torch.float32)
            components = {
                key: (numerator.detach(), denominator)
                for key in ("ce_loss", "l1_loss", "confidence_loss")
            }
            return (None, None, None, None, None, components, (numerator, denominator))

    reference = TorchReference.__new__(TorchReference)
    reference.model = Model()
    reference.documents = [None, None]
    reference.document_index = 0
    reference.denominator_sum = None
    reference.lm_head = torch.ones(1)
    calls = []
    reference.native_trainer_class = SimpleNamespace(
        _backward=lambda trainer, numerator, count: calls.append(count)
    )
    reference.native_trainer = SimpleNamespace()
    row = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "anchors": torch.zeros(1, 2, dtype=torch.long),
        "keep": torch.ones(1, 2, dtype=torch.bool),
    }
    reference.compute_document(row)
    reference.compute_document(row)
    assert calls == [2, 2]
    assert reference.model.sync == [False, True]
    assert reference.denominator_sum.dtype == torch.float32
    assert reference.denominator_sum.item() == 2.5


def test_common_state_equality_survives_checkpoint_helpers_consuming_the_mapping(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    source = torch.nn.Linear(2, 1)
    reference = TorchReference.__new__(TorchReference)
    reference.model = torch.nn.Linear(2, 1)
    reference.trainer = SimpleNamespace(model=source, dp_mesh=None)
    reference.optimizer = torch.optim.SGD(reference.model.parameters(), lr=0.1)
    module = ModuleType("torchspec.training.fsdp")

    def load(model, state, mesh, cpu_offload):
        model.load_state_dict(state)
        state["weight"] = object()  # Simulate replacing dense values with DTensors.
        return model

    module.fsdp2_load_full_state_dict = load
    monkeypatch.setitem(sys.modules, "torchspec.training.fsdp", module)
    reference.begin()
    assert reference.common_weights_equal


def test_comparison_keeps_common_state_and_independent_trajectories_separate(tmp_path):
    actual, reference = [], []
    for step in range(1, 17):
        common = {
            key: 2.0 for key in ("loss", "ce", "l1", "confidence", "denominator", "gradient_norm")
        }
        independent = {**common, "loss": 2.5}
        actual.append({"step": step, **common})
        reference.append(
            {
                "step": step,
                "common_weight": {**common, "gradient_relative_l2_diff": 0.01},
                "independent": independent,
                "common_weights_bitwise_equal": True,
                "documents": 4,
            }
        )
    for filename, rows in (("loss-steps", actual), ("torchspec-steps", reference)):
        (tmp_path / (filename + ".jsonl")).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    compare(tmp_path, 16)
    result = json.loads((tmp_path / "torchspec-comparison.json").read_text())
    assert result["common_weight"]["loss"]["max_relative_diff"] == 0
    assert result["independent"]["loss"]["max_relative_diff"] == 0.2
    assert result["compared_real_documents"] == 64
    (tmp_path / "torchspec-steps.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reference[:15]) + "\n"
    )
    with pytest.raises(ValueError, match="all sixteen"):
        compare(tmp_path, 16)
