# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_draft_model_config.py. Namespace bindings and ASpec regression coverage adapted.
# Additional cases adapted from AngelSpec 0966eb139d1c4054767e4a37bfc8a33db5989a72
# tests/test_draft_model_config.py. Tencent modifications are Apache-2.0;
# Copyright (C) 2026 Tencent. All rights reserved. See licenses/AngelSpec-LICENSE.

from types import SimpleNamespace

from aspec.config.utils import generate_draft_model_config
from aspec.utils import misc


def test_generate_draft_model_config_honors_remote_code_policy(monkeypatch):
    calls = []
    text_config = SimpleNamespace(
        vocab_size=8,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=128,
        rope_theta=10000,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )

    class DummyTokenizer:
        def __len__(self):
            return 8

    def load_config(*args, **kwargs):
        calls.append(("config", kwargs["trust_remote_code"]))
        return SimpleNamespace(text_config=text_config)

    def load_tokenizer(*args, **kwargs):
        calls.append(("tokenizer", kwargs["trust_remote_code"]))
        return DummyTokenizer()

    monkeypatch.setattr("aspec.config.utils.AutoConfig.from_pretrained", load_config)
    monkeypatch.setattr("aspec.config.utils.AutoTokenizer.from_pretrained", load_tokenizer)

    generate_draft_model_config("dummy-model")
    assert calls == [("config", False), ("tokenizer", False)]

    calls.clear()
    generate_draft_model_config("dummy-model", trust_remote_code=True)
    assert calls == [("config", True), ("tokenizer", True)]


def test_default_aux_layers_honor_remote_code_policy(monkeypatch):
    calls = []

    def load_config(*args, **kwargs):
        calls.append(kwargs["trust_remote_code"])
        return SimpleNamespace(num_hidden_layers=36)

    monkeypatch.setattr(misc.AutoConfig, "from_pretrained", load_config)

    assert misc.get_default_eagle3_aux_layer_ids("dummy-model") == [1, 17, 32]
    assert misc.get_default_eagle3_aux_layer_ids(
        "dummy-model", trust_remote_code=True
    ) == [1, 17, 32]
    assert calls == [False, True]


def test_generate_draft_model_config_preserves_rope_fields(monkeypatch):
    rope_scaling = {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "aspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "aspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")

    assert draft_config["max_position_embeddings"] == 262144
    assert draft_config["rope_theta"] == 5000000
    assert draft_config["rope_scaling"] == rope_scaling


def test_generate_draft_model_config_copies_rope_scaling(monkeypatch):
    rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=8192,
        max_position_embeddings=65536,
        rope_theta=1000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "aspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "aspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")
    rope_scaling["factor"] = 999.0

    assert draft_config["rope_scaling"]["factor"] == 8.0


def test_generate_draft_model_config_fills_yarn_defaults(monkeypatch):
    rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}
    text_config = SimpleNamespace(
        vocab_size=32000,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=8192,
        max_position_embeddings=65536,
        rope_theta=1000000,
        rope_scaling=rope_scaling,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    target_config = SimpleNamespace(text_config=text_config)

    class DummyTokenizer:
        def __len__(self):
            return 32000

    monkeypatch.setattr(
        "aspec.config.utils.AutoConfig.from_pretrained",
        lambda *args, **kwargs: target_config,
    )
    monkeypatch.setattr(
        "aspec.config.utils.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    draft_config = generate_draft_model_config("dummy-model")

    assert draft_config["rope_scaling"]["beta_fast"] == 32.0
    assert draft_config["rope_scaling"]["beta_slow"] == 1.0
    assert draft_config["rope_scaling"]["mscale"] == 1.0
    assert draft_config["rope_scaling"]["mscale_all_dim"] == 0.0
