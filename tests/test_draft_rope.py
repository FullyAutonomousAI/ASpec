# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests that a draft's rotary base is the one its config declares.

transformers 5.x moved `rope_theta` from a top-level config attribute into
`rope_parameters`, which silently reduced every affected draft to the library
default of 10000 while its config advertised something else. The sweep over
`configs/draft_models/` is the part that fails if a future release moves the
field again.
"""

import copy
import glob
import json
import os
import unittest
from types import SimpleNamespace

import pytest
import torch

from aspec.config.utils import resolve_rope_theta
from aspec.models.draft.auto import AutoDraftModelConfig
from aspec.models.draft.csa import _RotaryEmbedding
from aspec.models.draft.dflash import DFlashAttention, DFlashConfig
from aspec.models.draft.llama3_eagle import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMutiRotaryEmbedding,
    LlamaRotaryEmbedding,
    LlamaYarnRotaryEmbedding,
)
from aspec.models.draft.qsa import _PartialRotaryEmbedding

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT_CONFIG_DIR = os.path.join(REPO_ROOT, "aspec", "config")

def _declared_theta(raw):
    """The base a config file asks for, from wherever the file happens to put it."""
    block = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    nested = block.get("rope_theta")
    return nested if nested is not None else raw.get("rope_theta")


class TestResolveRopeTheta(unittest.TestCase):
    def test_prefers_top_level_attribute(self):
        cfg = DFlashConfig(rope_theta=50000.0)
        self.assertEqual(resolve_rope_theta(cfg), 50000.0)

    def test_falls_back_to_rope_parameters(self):
        # transformers 5.x absorbs a top-level rope_theta into rope_parameters,
        # leaving no attribute for the naive read to find.
        cfg = DFlashConfig(rope_theta=1000000.0)
        del cfg.rope_theta
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        self.assertIsNone(getattr(cfg, "rope_theta", None))
        self.assertEqual(resolve_rope_theta(cfg), 1000000.0)

    def test_default_when_declared_nowhere(self):
        cfg = DFlashConfig()
        del cfg.rope_theta
        self.assertEqual(resolve_rope_theta(cfg), 10000.0)
        self.assertIsNone(resolve_rope_theta(cfg, default=None))

    def test_conflicting_values_are_refused(self):
        cfg = DFlashConfig(rope_theta=10000.0)
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        with self.assertRaisesRegex(ValueError, "Conflicting rope_theta"):
            resolve_rope_theta(cfg)


class TestShippedDraftConfigs(unittest.TestCase):
    """Every config in configs/draft_models/ must resolve to the base it declares."""

    def test_declared_theta_is_what_resolves(self):
        paths = sorted(glob.glob(os.path.join(DRAFT_CONFIG_DIR, "*_draft_config.json")))
        self.assertGreater(len(paths), 0, "no draft configs found")
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                declared = _declared_theta(json.load(open(path)))
                if declared is None:
                    self.skipTest("config declares no rope_theta")
                cfg = AutoDraftModelConfig.from_file(path)
                self.assertEqual(resolve_rope_theta(cfg), float(declared))


class TestAttentionUsesResolvedTheta(unittest.TestCase):
    """Each rotary construction path must reach the resolved base, not the default."""

    def test_dflash_path(self):
        cfg = DFlashConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=1000000.0,
        )
        self.assertEqual(DFlashAttention(cfg).rotary_emb.base, 1000000.0)

    def test_eagle3_scaled_rope_keeps_configured_base(self):
        from aspec.models.draft.llama3_eagle import build_rotary_embedding

        for kind in ("linear", "dynamic"):
            with self.subTest(kind=kind):
                config = SimpleNamespace(rope_theta=1000000.0,
                                         rope_scaling={"rope_type": kind, "factor": 2.0})
                self.assertEqual(build_rotary_embedding(config, 16, 64).base, 1000000.0)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("cls", [_PartialRotaryEmbedding, _RotaryEmbedding])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtype_cast_preserves_rotary_frequencies_and_values(cls, dtype):
    torch.manual_seed(123)
    reference = cls(64, 40000.0)
    candidate = copy.deepcopy(reference).to(dtype=dtype)
    values = torch.randn(1, 4, 2, 64).to(dtype).requires_grad_()
    expected_values = values.detach().clone().requires_grad_()
    positions = torch.tensor([[0, 4095, 32767, 131071]])
    actual = candidate(values, positions)
    expected = reference(expected_values, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    expected.float().sum().backward()
    torch.testing.assert_close(values.grad, expected_values.grad, rtol=0, atol=0)
    assert candidate.inv_freq.dtype == torch.float32


@pytest.mark.parametrize("cls", [LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eagle_cache_growth_preserves_fp32_phases(cls, dtype):
    reference = cls(64, max_position_embeddings=64, base=40000.0)
    candidate = copy.deepcopy(reference).to(dtype)
    reference._set_cos_sin_cache(4096, torch.device("cpu"), dtype)
    candidate._set_cos_sin_cache(4096, torch.device("cpu"), dtype)
    torch.testing.assert_close(candidate.cos_cached, reference.cos_cached, rtol=0, atol=0)
    torch.testing.assert_close(candidate.sin_cached, reference.sin_cached, rtol=0, atol=0)

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eagle_mrope_preserves_fp32_phases(dtype):
    reference = LlamaMutiRotaryEmbedding(64, max_position_embeddings=64, base=40000.0)
    candidate = copy.deepcopy(reference).to(dtype)
    positions = torch.tensor([[[0, 4095, 32767, 131071]]]).expand(3, -1, -1)
    values = torch.zeros(1, 4, 64, dtype=dtype)
    for actual, expected in zip(candidate(values, positions), reference(values, positions), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["base", "linear", "dynamic", "llama3", "yarn"])
def test_rotary_cast_and_meta_materialization_preserve_scaled_definition(kind):
    factories = {
        "base": lambda: LlamaRotaryEmbedding(64, max_position_embeddings=64, base=40000.0),
        "linear": lambda: LlamaLinearScalingRotaryEmbedding(64, max_position_embeddings=64, base=40000.0, scaling_factor=2),
        "dynamic": lambda: LlamaDynamicNTKScalingRotaryEmbedding(64, max_position_embeddings=64, base=40000.0, scaling_factor=2),
        "llama3": lambda: LlamaRotaryEmbedding(64, max_position_embeddings=64, base=40000.0, scaling_factor=8, low_freq_factor=1, high_freq_factor=4, orig_max_position=8192),
        "yarn": lambda: LlamaYarnRotaryEmbedding(64, max_position_embeddings=64, base=40000.0, scaling_factor=2),
    }
    reference = factories[kind]()
    candidate = copy.deepcopy(reference).to(torch.bfloat16)
    torch.testing.assert_close(candidate.inv_freq, reference.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(candidate.cos_cached, reference.cos_cached.bfloat16(), rtol=0, atol=0)
    with torch.device("meta"):
        empty = factories[kind]()
    empty.to_empty(device="cpu")
    torch.testing.assert_close(empty.inv_freq, reference.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(empty.cos_cached, reference.cos_cached, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [_PartialRotaryEmbedding, _RotaryEmbedding])
def test_sparse_rotary_materializes_fp32_frequencies_from_meta(cls):
    reference = cls(64, 40000.0)
    with torch.device("meta"):
        candidate = cls(64, 40000.0)
    candidate.to_empty(device="cpu")
    torch.testing.assert_close(candidate.inv_freq, reference.inv_freq, rtol=0, atol=0)
