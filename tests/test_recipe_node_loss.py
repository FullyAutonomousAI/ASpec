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

from copy import deepcopy

import pytest

from tools.validation.recipe_node_loss import validate_loss


def valid():
    return [
        {"completed_step": 4},
        [{"NodeID": "worker", "Alive": True}],
        [{"NodeID": "worker", "Alive": False}],
        {
            "state": "failed",
            "observations": {
                "cleanup": {
                    "state": "completed",
                    "mode": "forced_owner_fence",
                    "work_lifecycle": {"resident_outputs": 0, "resident_bytes": 0},
                }
            },
        },
        {"optimizer_step": 4, "committed_work_ids": ["a", "b"]},
        10,
        300,
    ]


def test_node_loss_requires_dead_member_and_fenced_checkpoint():
    validate_loss(*valid())


@pytest.mark.parametrize(
    "case", ["unarmed", "alive", "healthy", "unfenced", "resident", "duplicate", "late", "absent"]
)
def test_node_loss_rejects_false_passes(case):
    args = deepcopy(valid())
    if case == "unarmed":
        args[0] = None
    elif case == "alive":
        args[2][0]["Alive"] = True
    elif case == "healthy":
        args[3]["state"] = "completed"
    elif case == "unfenced":
        args[3]["observations"]["cleanup"]["mode"] = "graceful"
    elif case == "resident":
        args[3]["observations"]["cleanup"]["work_lifecycle"]["resident_outputs"] = 1
    elif case == "duplicate":
        args[4]["committed_work_ids"] = ["a", "a"]
    elif case == "absent":
        args[2] = []
    else:
        args[5] = 301
    with pytest.raises(ValueError):
        validate_loss(*args)


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan")])
def test_node_loss_rejects_invalid_exit_budget(bound):
    args = valid()
    args[-1] = bound
    with pytest.raises(ValueError, match="finite"):
        validate_loss(*args)
