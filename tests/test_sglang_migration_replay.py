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

from types import SimpleNamespace

import torch

from tools.validation import sglang_migration_replay as replay


def test_full_state_statistics_do_not_build_autograd_graph(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    master = torch.nn.Parameter(parameter.detach().clone())
    parameter.grad = torch.tensor([0.2, 0.3])
    optimizer = SimpleNamespace(
        model_params=[parameter], fp32_params=[master],
        optimizer=SimpleNamespace(state={master: {
            "exp_avg": torch.tensor([0.1, 0.2]),
            "exp_avg_sq": torch.tensor([0.01, 0.04]),
        }}),
    )
    model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    original_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: original_zeros(
        *args, **{key: value for key, value in kwargs.items() if key != "device"}
    ))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    original_statistics = replay.add_statistics
    calls = []

    def statistics(total, reference, actual):
        assert not torch.is_grad_enabled()
        calls.append(actual.grad_fn)
        return original_statistics(total, reference, actual)

    monkeypatch.setattr(replay, "add_statistics", statistics)
    fields = {
        "model": parameter.detach().clone(), "master": master.detach().clone(),
        **optimizer.optimizer.state[master],
    }
    state = {"parameters": {"weight": fields}}
    result = replay.state_difference(model, optimizer, state, state, [master.detach().clone()])
    assert all(row["different_coordinates"] == 0 for row in result.values())
    assert replay.gradient_difference(model, optimizer, {"weight": parameter.grad})[
        "different_coordinates"
    ] == 0
    assert len(calls) == 6 and all(value is None for value in calls)
    assert torch.is_grad_enabled()
