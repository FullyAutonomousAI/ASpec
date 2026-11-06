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

import torch

from tools.validation.dspark_cp_operator_probe import difference


def test_operator_comparison_preserves_absolute_and_relative_vector_evidence():
    ref = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    delta, norm = difference(ref, ref.clone())
    assert delta == 0
    assert norm == 25
    delta, norm = difference(ref, torch.tensor([3.0, 5.0]))
    assert delta == 1
    assert norm == 25
    assert delta.dtype == norm.dtype == torch.float64
