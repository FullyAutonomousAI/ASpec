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
# tests/test_renderer_registry.py. Namespace bindings and ASpec regression coverage adapted.

from types import SimpleNamespace

import pytest
import torch

from aspec.data import dataset as dataset_module
from aspec.data.renderers import DEFAULT_CACHE_VERSION, RENDERER_REGISTRY
from aspec.data.utils import unpack_loss_mask

USER_HEADER = 101
ASSISTANT_HEADER = 102
END_OF_TURN = 103
MEDIA_PLACEHOLDER = 104


def _content_tokens(content):
    """Renderers see multimodal content unflattened, so handle both shapes."""
    if isinstance(content, str):
        return [ord(char) for char in content]
    tokens = []
    for item in content:
        if item.get("type") == "text":
            tokens.extend(ord(char) for char in item["text"])
        else:
            tokens.append(MEDIA_PLACEHOLDER)
    return tokens


class StubTokenizer:
    """Records what the renderer was handed so tests can assert plumbing."""

    def __init__(self):
        self.calls = []


class StubRenderer:
    """Character-level renderer: one token per content character."""

    CACHE_VERSION = "stub-v3"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_assistant_token_ids(self):
        return [ASSISTANT_HEADER], [END_OF_TURN], 0

    def render(
        self,
        messages,
        tools=None,
        *,
        max_seq_length,
        last_turn_only=False,
        generation_config=None,
    ):
        self.tokenizer.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_seq_length": max_seq_length,
                "last_turn_only": last_turn_only,
                "generation_config": generation_config,
            }
        )
        assistant_indices = [
            index for index, msg in enumerate(messages) if msg["role"] == "assistant"
        ]
        supervised = set(assistant_indices[-1:] if last_turn_only else assistant_indices)

        input_ids, loss_mask = [], []
        for index, msg in enumerate(messages):
            is_assistant = msg["role"] == "assistant"
            input_ids.append(ASSISTANT_HEADER if is_assistant else USER_HEADER)
            loss_mask.append(0)
            body = _content_tokens(msg["content"])
            input_ids.extend(body)
            loss_mask.extend([1 if index in supervised else 0] * len(body))
            input_ids.append(END_OF_TURN)
            loss_mask.append(0)
        return input_ids[:max_seq_length], loss_mask[:max_seq_length]


class MinimalRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_assistant_token_ids(self):
        return [ASSISTANT_HEADER], [END_OF_TURN], 0

    def render(self, messages, tools=None, **_kwargs):
        return [1], [1]


@pytest.fixture
def stub_renderer():
    RENDERER_REGISTRY.register("stub", StubRenderer, override=True)
    try:
        yield StubRenderer
    finally:
        RENDERER_REGISTRY.renderers.pop("stub", None)


def _body_indices(input_ids, text):
    body = [ord(char) for char in text]
    for start in range(len(input_ids) - len(body) + 1):
        if input_ids[start : start + len(body)] == body:
            return list(range(start, start + len(body)))
    raise AssertionError(f"{text!r} not found in rendered token ids")


def _messages():
    return [
        {"role": "user", "content": "ask one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "ask two"},
        {"role": "assistant", "content": "second"},
    ]


def test_registry_creates_renderers_with_the_target_tokenizer(stub_renderer):
    tokenizer = StubTokenizer()

    renderer = RENDERER_REGISTRY.create("stub", tokenizer)

    assert isinstance(renderer, StubRenderer)
    assert renderer.tokenizer is tokenizer
    assert "stub" in RENDERER_REGISTRY.get_all_renderer_names()


def test_registry_rejects_duplicate_registration(stub_renderer):
    with pytest.raises(ValueError, match="already been registered"):
        RENDERER_REGISTRY.register("stub", StubRenderer)


def test_cache_version_falls_back_to_the_protocol_default(stub_renderer):
    RENDERER_REGISTRY.register("stub-minimal", MinimalRenderer, override=True)
    try:
        assert RENDERER_REGISTRY.cache_version("stub") == "stub-v3"
        assert RENDERER_REGISTRY.cache_version("stub-minimal") == DEFAULT_CACHE_VERSION
    finally:
        RENDERER_REGISTRY.renderers.pop("stub-minimal", None)


def test_tokenize_worker_forwards_tools_and_generation_config(stub_renderer, monkeypatch):
    tokenizer = StubTokenizer()
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: tokenizer)
    dataset_module._worker_state.clear()
    dataset_module._init_tokenize_worker(
        "stub-model",
        True,
        None,
        last_turn_loss_only=True,
        min_loss_tokens=1,
        renderer_name="stub",
    )
    tools = [{"type": "function", "function": {"name": "search"}}]
    generation_config = {"temperature": 0.7}

    result = dataset_module._tokenize_single((_messages(), tools, generation_config, 1024, False))

    assert result is not None
    assert result["formatted_prompt"] is None
    assert tokenizer.calls[-1]["tools"] is tools
    assert tokenizer.calls[-1]["generation_config"] is generation_config
    assert tokenizer.calls[-1]["last_turn_only"] is True

    input_ids = result["input_ids"].tolist()
    loss_mask = unpack_loss_mask(result["packed_loss_mask"]).tolist()
    assert len(loss_mask) == len(input_ids)
    assert all(loss_mask[i] == 0 for i in _body_indices(input_ids, "first"))
    assert all(loss_mask[i] == 1 for i in _body_indices(input_ids, "second"))


def test_tokenize_worker_rejects_train_with_decode(stub_renderer, monkeypatch):
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: StubTokenizer())
    dataset_module._worker_state.clear()
    dataset_module._init_tokenize_worker("stub-model", True, None, renderer_name="stub")

    with pytest.raises(ValueError, match="train_with_decode"):
        dataset_module._tokenize_single((_messages(), None, None, 1024, True))


def test_tokenize_worker_skips_samples_with_no_supervision(stub_renderer, monkeypatch):
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: StubTokenizer())
    dataset_module._worker_state.clear()
    dataset_module._init_tokenize_worker("stub-model", True, None, renderer_name="stub")
    prompt_only = [{"role": "user", "content": "ask one"}]

    assert dataset_module._tokenize_single((prompt_only, None, None, 1024, False)) is None


def _renderer_args(tmp_path, source, **overrides):
    args = dict(
        cache_dir=str(tmp_path / "cache"),
        chat_template=None,
        defer_tokenization=False,
        last_turn_loss_only=True,
        max_seq_length=1024,
        min_loss_tokens=1,
        num_proc=1,
        prompt_key="conversations",
        renderer="stub",
        target_model_path="stub-model",
        train_data_path=str(source),
        train_with_decode=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_dataset_loading_routes_sharegpt_rows_through_the_renderer(
    stub_renderer, tmp_path, monkeypatch
):
    tokenizer = StubTokenizer()
    source = tmp_path / "rows.jsonl"
    source.write_text("{}\n")
    sample = {
        "data_id": "row-1",
        "conversations": [
            {"from": "human", "value": "ask one"},
            {"from": "gpt", "value": "first"},
            {"from": "human", "value": "ask two"},
            {"from": "gpt", "value": "second"},
        ],
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "generation_config": {"temperature": 0.2},
    }
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: [sample])
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: tokenizer)

    admission = {}
    prompts = dataset_module.load_conversation_dataset(
        _renderer_args(tmp_path, source), admission_report=admission
    )

    assert len(prompts) == 1
    assert prompts[0]["data_id"] == "row-1"
    assert prompts[0]["formatted_prompt"] is None
    assert tokenizer.calls[-1]["tools"] == sample["tools"]
    assert tokenizer.calls[-1]["generation_config"] == sample["generation_config"]
    input_ids = prompts[0]["input_ids"].tolist()
    loss_mask = unpack_loss_mask(prompts[0]["packed_loss_mask"]).tolist()
    assert all(loss_mask[i] == 0 for i in _body_indices(input_ids, "first"))
    assert all(loss_mask[i] == 1 for i in _body_indices(input_ids, "second"))
    assert admission == {
        "state": "admitted",
        "source_samples": 1,
        "processed_samples": 1,
        "admitted_samples": 1,
        "skipped_samples": 0,
        "unbalanced_thinking_samples": 0,
        "cache_state": "miss",
    }

    cached_admission = {}
    cached = dataset_module.load_conversation_dataset(
        _renderer_args(tmp_path, source), admission_report=cached_admission
    )
    assert len(cached) == 1
    assert cached_admission == {**admission, "cache_state": "hit"}


def test_dataset_worker_initializer_failure_prevents_pool_restart_loop(
    stub_renderer, tmp_path, monkeypatch
):
    source = tmp_path / "rows.jsonl"
    source.write_text("{}\n")
    sample = {
        "data_id": "row-1",
        "conversations": [
            {"from": "human", "value": "ask"},
            {"from": "gpt", "value": "answer"},
        ],
    }
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: [sample])
    monkeypatch.setattr(
        dataset_module,
        "load_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing tokenizer")),
    )

    def unexpected_pool(*_args, **_kwargs):
        raise AssertionError("pool must not start after initializer preflight fails")

    monkeypatch.setattr(dataset_module.mp, "Pool", unexpected_pool)
    args = _renderer_args(tmp_path, source, num_proc=4)

    with pytest.raises(RuntimeError, match="failed before starting.*multiprocessing pool") as exc:
        dataset_module.load_conversation_dataset(args)

    assert isinstance(exc.value.__cause__, OSError)


def test_dataset_loading_rejects_an_unknown_renderer(tmp_path):
    args = _renderer_args(tmp_path, tmp_path / "rows.jsonl", renderer="no-such-renderer")

    with pytest.raises(ValueError, match="Unknown dataset renderer"):
        dataset_module.load_conversation_dataset(args)


def test_dataset_loading_requires_a_renderer_or_a_chat_template(tmp_path):
    # Raw conversations only: a pretokenized corpus needs neither setting, so
    # the requirement can only be enforced once the schema is known.
    source = tmp_path / "rows.jsonl"
    source.write_text("{}\n")
    args = _renderer_args(tmp_path, source, renderer=None)

    with pytest.raises(ValueError, match="Either renderer or chat_template"):
        dataset_module.load_conversation_dataset(args)


def test_renderer_rejects_deferred_tokenization(stub_renderer, tmp_path):
    args = _renderer_args(tmp_path, tmp_path / "rows.jsonl", defer_tokenization=True)

    with pytest.raises(ValueError, match="defer_tokenization=False"):
        dataset_module.load_conversation_dataset(args)


def _multimodal_sample():
    return {
        "data_id": "row-mm",
        "conversations": [
            {
                "from": "human",
                "value": [
                    {"type": "image", "image": "example.png"},
                    {"type": "text", "text": "describe"},
                ],
            },
            {"from": "gpt", "value": "answer"},
        ],
    }


def test_multimodal_rows_defer_their_mask_to_the_training_matcher(
    stub_renderer, tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: [_multimodal_sample()])
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: StubTokenizer())
    source = tmp_path / "rows-mm.jsonl"
    source.write_text("{}\n")

    prompts = dataset_module.load_conversation_dataset(
        _renderer_args(tmp_path, source, dynamic_loss_mask=True)
    )

    assert len(prompts) == 1
    assert "input_ids" in prompts[0]
    assert "packed_loss_mask" not in prompts[0]
    assert prompts[0]["multimodal_inputs"] == {"images": ["example.png"], "videos": None}


def test_multimodal_rows_fail_closed_without_a_dynamic_mask(stub_renderer, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: [_multimodal_sample()])
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: StubTokenizer())
    source = tmp_path / "rows-mm.jsonl"
    source.write_text("{}\n")

    with pytest.raises(ValueError, match="dynamic_loss_mask=True"):
        dataset_module.load_conversation_dataset(_renderer_args(tmp_path, source))


def test_cached_multimodal_rows_drop_their_pre_expansion_mask():
    prompts = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "packed_loss_mask": "2,1",
            "multimodal_inputs": {"images": ["example.png"]},
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "packed_loss_mask": "1,2",
            "multimodal_inputs": None,
        },
    ]

    dataset_module._drop_stale_multimodal_masks(prompts, dynamic_loss_mask=True)

    assert "packed_loss_mask" not in prompts[0]
    assert prompts[1]["packed_loss_mask"] == "1,2"


def test_cached_masks_survive_when_dynamic_masking_is_off():
    prompts = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "packed_loss_mask": "2,1",
            "multimodal_inputs": {"images": ["example.png"]},
        }
    ]

    dataset_module._drop_stale_multimodal_masks(prompts, dynamic_loss_mask=False)

    assert prompts[0]["packed_loss_mask"] == "2,1"


def test_assistant_matcher_comes_from_the_renderer(stub_renderer, monkeypatch):
    from aspec.utils import processing

    monkeypatch.setattr(processing, "load_tokenizer", lambda *_a, **_kw: StubTokenizer())
    args = SimpleNamespace(renderer="stub", chat_template="llama3", target_model_path="stub-model")

    header_ids, end_ids, skip_after_header = processing.get_assistant_token_ids(args)

    assert header_ids == [ASSISTANT_HEADER]
    assert end_ids == [END_OF_TURN]
    assert skip_after_header == 0


def test_renderer_config_enables_the_dynamic_loss_mask():
    from aspec.config.train_config import config_to_flat_args, load_config

    args = config_to_flat_args(load_config(cli_args=["dataset.renderer=stub"]))

    assert args.renderer == "stub"
    assert args.dynamic_loss_mask is True
    assert args.defer_tokenization is False


def test_renderer_config_rejects_offline_replay(tmp_path):
    from aspec.config.train_config import load_config

    with pytest.raises(ValueError, match="recorded loss masks"):
        load_config(
            cli_args=[
                "dataset.renderer=stub",
                "inference.inference_engine_type=offline",
                f"inference.offline.data_path={tmp_path}",
            ]
        )

@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("renderer", [None, "stub"])
def test_assistant_matcher_preserves_remote_code_policy(stub_renderer, monkeypatch, trust, renderer):
    from aspec.utils import processing
    calls = []
    def load(name, **kwargs):
        calls.append(kwargs["trust_remote_code"])
        tokenizer = StubTokenizer()
        tokenizer.encode = lambda text, **kwargs: [ord(char) for char in text]
        return tokenizer
    monkeypatch.setattr(processing, "load_tokenizer", load)
    args = SimpleNamespace(renderer=renderer, chat_template="llama3", target_model_path="stub-model",
                           trust_remote_code=trust)
    processing.get_assistant_token_ids(args)
    assert calls == [trust]


def test_failed_tokenization_cache_write_is_not_published(stub_renderer, tmp_path, monkeypatch):
    source = tmp_path / "rows.jsonl"
    source.write_text("{}\n")
    sample = {"data_id": "row-1", "conversations": _messages()}
    tokenizer = StubTokenizer()
    monkeypatch.setattr(dataset_module, "load_hf_dataset", lambda _path: [sample])
    monkeypatch.setattr(dataset_module, "load_tokenizer", lambda *_a, **_kw: tokenizer)
    args = _renderer_args(tmp_path, source)
    cache_dir = tmp_path / "cache" / "tokenized_dataset"

    def interrupted_save(_payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"incomplete")
        else:
            from pathlib import Path
            Path(destination).write_bytes(b"incomplete")
        raise OSError("simulated interrupted cache write")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(dataset_module.torch, "save", interrupted_save)
        with pytest.raises(OSError, match="interrupted cache write"):
            dataset_module.load_conversation_dataset(args)
    assert list(cache_dir.iterdir()) == []

    admission = {}
    prompts = dataset_module.load_conversation_dataset(args, admission_report=admission)
    assert admission["cache_state"] == "miss"
    cached = dataset_module.load_conversation_dataset(args, admission_report=admission)
    assert admission["cache_state"] == "hit"
    assert torch.equal(prompts[0]["input_ids"], cached[0]["input_ids"])
    assert prompts[0]["packed_loss_mask"] == cached[0]["packed_loss_mask"]
