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

"""``save_pretrained`` / ``from_pretrained`` round trip for every draft family.

No draft calls ``post_init()``, where Transformers 5.x populates
``all_tied_weights_keys``, so without the base classes declaring it empty every
one of these loads dies with an ``AttributeError``.
"""

import tempfile
import unittest

import torch

from aspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.base import Eagle3DraftModel
from aspec.models.draft.dflash import DFlashDraftModel

_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "dflash",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_target_layers": 2,
    "target_hidden_size": 64,
    "target_num_hidden_layers": 12,
    "mask_token_id": 127,
}

_DSPARK = {
    **_DFLASH,
    "architectures": ["Qwen3DSparkModel"],
    "model_type": "qwen3_dspark",
    "markov_rank": 16,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
}

DRAFT_FAMILIES = {
    "dflash": _DFLASH,
    "dspark": _DSPARK,
}


def _build(config_dict):
    config = AutoDraftModelConfig.from_dict(config_dict)
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


class TestTiedWeightsKeysDeclared(unittest.TestCase):
    def test_base_classes_declare_an_empty_mapping(self):
        self.assertEqual(Eagle3DraftModel.all_tied_weights_keys, {})
        self.assertEqual(DFlashDraftModel.all_tied_weights_keys, {})

    def test_every_family_exposes_the_mapping(self):
        for name, config_dict in DRAFT_FAMILIES.items():
            with self.subTest(family=name):
                model = _build(config_dict)
                self.assertEqual(model.all_tied_weights_keys, {})


class TestFromPretrainedRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_every_parameter(self):
        for name, config_dict in DRAFT_FAMILIES.items():
            with self.subTest(family=name):
                model = _build(config_dict)
                with tempfile.TemporaryDirectory() as tmpdir:
                    model.save_pretrained(tmpdir)
                    reloaded = AutoEagle3DraftModel.from_pretrained(
                        tmpdir, config=model.config, torch_dtype=torch.float32
                    )

                original = model.state_dict()
                restored = reloaded.state_dict()
                self.assertEqual(sorted(original), sorted(restored))
                for key, value in original.items():
                    self.assertTrue(
                        torch.equal(value, restored[key]),
                        msg=f"{name}: {key} changed across the round trip",
                    )

if __name__ == "__main__":
    unittest.main()


def test_pretrained_loader_preserves_formatted_upstream_warnings(monkeypatch, caplog):
    from transformers import AutoModelForCausalLM, modeling_utils

    result = object()
    def load(cls, path, *args, **kwargs):
        modeling_utils.logger.warning("Loading checkpoint %s", path)
        return result
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", classmethod(load))
    assert AutoEagle3DraftModel.from_pretrained("local-draft") is result
    assert "Loading checkpoint local-draft" in caplog.text
