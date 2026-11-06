# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.

"""Checkpoint-local L1 statistics reuse preserves native head gradients."""

import copy
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires the SM90 loss-kernel integration environment",
)


@pytest.mark.parametrize("indexed", [False, True])
def test_checkpoint_reuses_only_small_statistics_with_identical_head_gradients(monkeypatch, indexed):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel
    from aspec.models.ops import tiled_l1

    torch.manual_seed(37)
    config = DSparkConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                         vocab_size=8192, mask_token_id=8191, markov_rank=16,
                         num_target_layers=1, target_hidden_size=64,
                         target_num_hidden_layers=4, target_layer_ids=[1])
    actual = DSparkModel(DSparkDraftModel(config)).cuda().bfloat16()
    reference = copy.deepcopy(actual)
    head = torch.randn(8192, 64, device="cuda", dtype=torch.bfloat16) * 0.03
    hidden = torch.randn(2, 8, 7, 64, device="cuda", dtype=torch.bfloat16)
    teacher = torch.randn(2, 97, 8192, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(97, hidden.shape[:-1], device="cuda")
    positions[:, :2] = 0
    labels = torch.randint(8192, positions.shape, device="cuda")
    mask = (torch.rand(positions.shape, device="cuda") > 0.2).float()
    weights = mask * torch.exp(-torch.arange(7, device="cuda").float() / 4)
    if not indexed:
        teacher = teacher.gather(1, positions.flatten(1)[..., None].expand(-1, -1, 8192)).view(2, 8, 7, 8192)
    original_stats = tiled_l1._stats.run
    launches = []

    def stats(*args, **kwargs):
        launches.append(1)
        return original_stats(*args, **kwargs)

    monkeypatch.setattr(tiled_l1._stats, "run", stats)
    # Fresh cache for every update; it is not model/checkpoint state.
    for step in range(2):
        outputs, gradients, counts = [], [], []
        for model in (actual, reference):
            model.zero_grad(set_to_none=True)
            value = (hidden + step * 0.1).detach().requires_grad_()
            cache = {} if model is actual else None
            before = len(launches)
            terms = checkpoint(
                model._prediction_terms, value, labels, labels, head, torch.zeros_like(value),
                mask, weights, teacher, None, positions if indexed else None, cache,
                use_reentrant=False, preserve_rng_state=False,
            )
            ((0.1 * terms[0] + 0.9 * terms[1] + terms[2]) * 0.017).backward()
            outputs.append(terms)
            gradients.append([value.grad] + [p.grad for p in model.parameters() if p.grad is not None])
            counts.append(len(launches) - before)
            if cache is not None:
                tensors = cache["statistics"]
                assert sum(t.numel() * t.element_size() for t in tensors) == 32 * positions.numel()
                assert all(not t.requires_grad and t.grad_fn is None for t in tensors)
        assert counts == [1, 2]
        for candidate, old in zip(*outputs, strict=True):
            torch.testing.assert_close(candidate, old, atol=0, rtol=0)
        for candidate, old in zip(*gradients, strict=True):
            torch.testing.assert_close(candidate, old, atol=0, rtol=0)
