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

import json
from unittest.mock import Mock

import pytest

from tools.evaluate_draft import load_prompts, output_checks, speculative_token_budget, summarize


def test_decode_prompt_does_not_contain_held_out_answer(tmp_path):
    data = tmp_path / "data.jsonl"
    messages = [{"role": "user", "content": "question"},
                {"role": "assistant", "content": "secret answer"}]
    data.write_text(json.dumps({"conversations": messages}) + "\n")
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2]
    assert load_prompts(data, tokenizer, 1) == [[1, 2]]
    assert tokenizer.apply_chat_template.call_args.args[0] == messages[:1]
    with pytest.raises(ValueError, match="found 1"):
        load_prompts(data, tokenizer, 2)


def test_acceptance_is_weighted_by_actual_verifications_not_mean_of_means():
    rows = [[{"wall_seconds": 1, "meta_info": {"completion_tokens": 10, "spec_verify_ct": 2}},
             {"wall_seconds": 3, "meta_info": {"completion_tokens": 12, "spec_verify_ct": 6}}]]
    result = summarize(rows)
    assert result["acceptance_length"] == 22 / 8
    assert result["repeat_tokens_per_second"] == [22 / 4]


def test_speculative_evaluation_cannot_claim_enablement_without_verifier_counts():
    rows = [[{"wall_seconds": 1, "meta_info": {"completion_tokens": 10}}]]
    assert summarize(rows)["acceptance_length"] is None  # Target-only is valid.
    with pytest.raises(ValueError, match="actual verifier-count evidence"):
        summarize(rows, require_verification=True)


def test_decode_uses_saved_generation_prefix_not_training_answer(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"input_ids": [1, 2, 99], "loss_mask": [0, 0, 1],
                                "prompt_input_ids": [1, 2]}) + "\n")
    assert load_prompts(data, Mock(), 1) == [[1, 2]]
    data.write_text(json.dumps({"input_ids": [1, 2, 99], "loss_mask": [0, 0, 1]}) + "\n")
    with pytest.raises(ValueError, match="exclude held-out answers"):
        load_prompts(data, Mock(), 1)


def test_decode_non_thinking_is_explicit_and_old_default_is_preserved(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "question"}]}) + "\n")
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2]
    load_prompts(data, tokenizer, 1, enable_thinking=False)
    assert tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    load_prompts(data, tokenizer, 1)
    assert "enable_thinking" not in tokenizer.apply_chat_template.call_args.kwargs


def test_generated_decode_selection_is_independent_of_completion_order(tmp_path):
    data = tmp_path / "data.jsonl"
    rows = [{"prompt_input_ids": [i], "input_ids": [i, 99], "loss_mask": [0, 1]}
            for i in range(1, 9)]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = load_prompts(data, Mock(), 3)
    data.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    assert load_prompts(data, Mock(), 3) == selected


def test_dspark_verify_budget_includes_the_target_token_not_an_extra_training_slot():
    assert speculative_token_budget({"block_size": 7}, "DSPARK") == 8
    assert speculative_token_budget({"block_size": 11}, "DSPARK") == 12
    assert speculative_token_budget({}, "DSPARK") == 8  # Legacy default gamma=7.
    assert speculative_token_budget({"block_size": 7}, "DFLASH") == 7
    with pytest.raises(ValueError, match="block_size"):
        speculative_token_budget({"block_size": True}, "DSPARK")
    with pytest.raises(ValueError, match="Conflicting"):
        speculative_token_budget({"block_size": 7, "dflash_config": {"block_size": 8}}, "DSPARK")


def test_equality_between_frameworks_does_not_hide_repeat_instability():
    repeats = [[{"output_ids": [1]}], [{"output_ids": [2]}]]
    verdict = output_checks(repeats, repeats)
    assert verdict["greedy_equal"]
    assert not verdict["target_repeat_stable"]
    assert not verdict["draft_repeat_stable"]
    stable = [repeats[0], repeats[0]]
    assert all(output_checks(stable, stable).values())


@pytest.mark.parametrize("baseline,candidate", [([], []), ([[]], [[]]),
    ([[{"output_ids": [1]}]], []), ([[{"output_ids": [1]}]], [[]])])
def test_empty_or_incomplete_repeats_cannot_pass_equality(baseline, candidate):
    with pytest.raises(ValueError, match="complete decode repeats"):
        output_checks(baseline, candidate)
