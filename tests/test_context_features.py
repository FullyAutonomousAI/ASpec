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

import copy

import pytest
import torch

from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


@pytest.mark.parametrize("fc_norm", [False, True])
def test_concatenated_features_match_layer_views_and_gradients(fc_norm):
    torch.manual_seed(42)
    config = DFlashConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        num_target_layers=3,
        target_hidden_size=48,
        target_num_hidden_layers=12,
        fc_norm=fc_norm,
    )
    reference = DFlashDraftModel(config)
    candidate = copy.deepcopy(reference)
    source = torch.randn(2, 128, 144, requires_grad=True)
    candidate_source = source.detach().clone().requires_grad_()
    reference_output = reference.extract_context_feature(list(source.split(48, dim=-1)))
    seen = []
    hook = candidate.context_proj.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].data_ptr())
    )
    candidate_output = candidate.extract_context_feature(candidate_source)
    hook.remove()
    torch.testing.assert_close(candidate_output, reference_output)
    upstream = torch.randn_like(reference_output)
    reference_output.backward(upstream)
    candidate_output.backward(upstream)
    torch.testing.assert_close(candidate_source.grad, source.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        reference_grad = reference_parameters[name].grad
        if reference_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference_grad)
    if not fc_norm:
        assert seen == [candidate_source.data_ptr()]
