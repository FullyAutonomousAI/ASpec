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

from aspec.models.ops.direct_cp import DirectCPMetadata, _math_partial
from tools.validation.dspark_fa4_numerics import (
    analytic_partial,
    delta_lse_correction,
    vector_stats,
)


@pytest.mark.parametrize("with_lse", [False, True])
def test_analytic_gqa_gradient_matches_independent_autograd(with_lse):
    torch.manual_seed(91)
    q = torch.randn(1, 4, 8, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 17, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 17, 4, dtype=torch.float64, requires_grad=True)
    metadata = DirectCPMetadata(
        torch.tensor([[0, 4, 7, 8]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.arange(9)[None],
        2,
        0,
        1,
        None,
        backend="math",
    )
    output, lse = _math_partial(q, k, v, metadata, 0.5)
    dout = torch.randn_like(output)
    dlse = torch.randn_like(lse) if with_lse else None
    objective = (output * dout).sum()
    if with_lse:
        objective = objective + (lse * dlse).sum()
    objective.backward()
    actual_output, gradients = analytic_partial(q, k, v, metadata, 0.5, dout, dlse)
    torch.testing.assert_close(actual_output, output)
    for expected, actual in zip((q.grad, k.grad, v.grad), gradients):
        torch.testing.assert_close(actual, expected)


def test_lse_correction_changes_only_delta_not_upstream_output_gradient():
    exact = torch.tensor([[[[5.231, -7.789]]]], dtype=torch.float64)
    rounded = exact.bfloat16()
    dout = torch.tensor([[[[0.011, -0.071]]]], dtype=torch.bfloat16)
    dlse = torch.tensor([[[0.093]]])
    original_dout = dout.clone()
    correction = delta_lse_correction(rounded, exact, dout)
    corrected_delta = (rounded.double() * dout.double()).sum(-1) - (dlse + correction).double()
    exact_delta = (exact * dout.double()).sum(-1) - dlse.double()
    torch.testing.assert_close(corrected_delta, exact_delta, atol=1e-8, rtol=1e-8)
    assert torch.equal(dout, original_dout)


def test_vector_stats_does_not_hide_absolute_error_or_direction():
    stats = vector_stats(torch.tensor([3e-4, 4e-4]), torch.tensor([3e-4, -4e-4]))
    assert stats["reference_l2"] == pytest.approx(5e-4)
    assert stats["absolute_l2"] == pytest.approx(8e-4)
    assert stats["relative_l2"] == pytest.approx(1.6)
    assert stats["cosine"] == pytest.approx(-0.28)
