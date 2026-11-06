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
# tests/test_conversation_normalization.py. Namespace bindings and ASpec regression coverage adapted.

from copy import deepcopy

from aspec.data.preprocessing import _normalize_conversation


def test_sharegpt_normalization_preserves_extended_message_fields():
    conversation = [
        {"from": "human", "value": "Look up x.", "name": "requester"},
        {
            "from": "gpt",
            "value": "",
            "thinking": "I should call the lookup tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"x"}',
                    },
                }
            ],
        },
        {
            "from": "tool",
            "value": "42",
            "tool_call_id": "call_1",
            "name": "lookup",
        },
    ]
    original = deepcopy(conversation)

    normalized = _normalize_conversation(conversation)

    assert conversation == original
    assert normalized == [
        {"role": "user", "content": "Look up x.", "name": "requester"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the lookup tool.",
            "tool_calls": original[1]["tool_calls"],
        },
        {
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_1",
            "name": "lookup",
        },
    ]


def test_sharegpt_role_and_content_win_over_stale_duplicates():
    conversation = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi", "role": "system", "content": "stale"},
    ]

    assert _normalize_conversation(conversation) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_reasoning_aliases_collapse_to_reasoning_content():
    conversation = [
        {"from": "gpt", "value": "answer", "thinking_content": "why"},
    ]

    normalized = _normalize_conversation(conversation)

    assert normalized == [{"role": "assistant", "content": "answer", "reasoning_content": "why"}]
    assert "thinking_content" not in normalized[0]


def test_standard_conversation_is_left_unchanged():
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert _normalize_conversation(conversation) is conversation
