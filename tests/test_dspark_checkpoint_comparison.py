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
import torch.distributed.checkpoint as dcp

from tools.validation.dspark_checkpoint_comparison import inspect_checkpoint


def save_fixture(root, *, step=32, wrong_model=False):
    master = torch.tensor([1.002, 2.007], dtype=torch.float32)
    model = master.to(torch.bfloat16) + int(wrong_model)
    dcp.save(
        {"model_state": {"model": {"draft_model.layer.weight": model}}},
        checkpoint_id=str(root / "model"),
    )
    dcp.save(
        {
            "optim_state": {
                "fp32_params": {"layer.weight": master},
                "optim": {
                    "state": {
                        0: {
                            "step": torch.tensor(float(step)),
                            "exp_avg": master * 0.1,
                            "exp_avg_sq": master * 0.01,
                        }
                    }
                },
            }
        },
        checkpoint_id=str(root / "optimizer"),
    )
    return master


def test_actual_checkpoint_names_and_master_rounding_are_checked(tmp_path):
    expected = save_fixture(tmp_path)
    masters, model, counts = inspect_checkpoint(tmp_path)
    assert torch.equal(masters["layer.weight"], expected)
    assert torch.equal(model["layer.weight"], expected.to(torch.bfloat16))
    assert counts == [32]


@pytest.mark.parametrize(
    "change, message",
    [({"step": 31}, "Adam frontier"), ({"wrong_model": True}, "optimizer master")],
)
def test_incomplete_frontier_and_stale_model_fail_closed(tmp_path, change, message):
    save_fixture(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        inspect_checkpoint(tmp_path)
