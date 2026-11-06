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
import hashlib
import json

import pytest

from tools.validation.dspark_quality_comparison import (
    question_comparisons,
    real_acceptance_comparison,
    training_summary,
)


def decode_fixture(verifies=(2, 4)):
    repeats = [
        [
            {
                "output_ids": list(range(tokens)),
                "meta_info": {"completion_tokens": tokens, "spec_verify_ct": verify},
            }
            for tokens, verify in zip((4, 8), verifies)
        ]
        for _ in range(3)
    ]
    return {
        "algorithm": "DSPARK",
        "target": "qwen",
        "versions": {"sglang": "0.5.21"},
        "prompt_tokens_sha256": "prompts",
        "data_sha256": "data",
        "settings": {
            "samples": 2,
            "repeats": 3,
            "max_new_tokens": 8,
            "tp": 1,
            "temperature": 0,
            "disable_thinking": True,
            "deterministic_inference": True,
        },
        "runs": {
            mode: {
                "samples": copy.deepcopy(repeats),
                "effective_engine_args": {"num_draft_tokens": 8},
            }
            for mode in ("target_only", "speculative")
        },
    }


def test_real_acceptance_uses_verifier_totals_and_independent_questions():
    result = real_acceptance_comparison(decode_fixture((1, 2)), decode_fixture(), bootstrap=100)
    assert result["questions"] == 2 and result["repeats"] == 3
    assert result["acceptance_length"] == {"aspec": 4.0, "torchspec": 2.0}
    assert result["relative_difference"] == 1.0
    assert result["paired_question_bootstrap_95ci"]["difference"] == [2.0, 2.0]
    assert result["actual_counters"]["aspec"] == {"completion_tokens": 36, "verifier_calls": 9}


@pytest.mark.parametrize("failure", ("prompt", "effective", "output", "repeat", "counter"))
def test_decode_protocol_or_output_mismatch_cannot_be_reported_as_quality(failure):
    fast, reference = decode_fixture(), decode_fixture()
    if failure == "prompt":
        fast["prompt_tokens_sha256"] = "other"
    elif failure == "effective":
        fast["runs"]["speculative"]["effective_engine_args"]["num_draft_tokens"] = 7
    elif failure == "output":
        fast["runs"]["speculative"]["samples"][0][0]["output_ids"] = [99]
    elif failure == "repeat":
        fast["runs"]["target_only"]["samples"].pop()
    else:
        for repeat in fast["runs"]["speculative"]["samples"]:
            for row in repeat:
                row["meta_info"]["spec_verify_ct"] = 0
    with pytest.raises(ValueError):
        real_acceptance_comparison(fast, reference, bootstrap=100)


def test_training_summary_keeps_resume_frontier_and_selection_rule(tmp_path):
    prefix, continuation = tmp_path / "prefix", tmp_path / "continuation"
    terms = {"loss": 2.0, "ce": 3.0, "l1": 1.0, "confidence": 0.8, "denominator": 10.0}
    for step, root in enumerate((prefix, continuation), 1):
        root.mkdir()
        record = {
            "step": step,
            "learning_rate_before": 0.0006,
            "aspec": terms,
            "torchspec": terms,
        }
        for rank in range(4):
            (root / f"steps-rank{rank}.jsonl").write_text(json.dumps(record) + "\n")
        validation = {"step": step, "aspec": {**terms, "ce": 4 - step}, "torchspec": terms}
        for rank in range(4):
            (root / f"validation-rank{rank}.jsonl").write_text(json.dumps(validation) + "\n")
    result, curve = training_summary([prefix, continuation], 2)
    assert result["documents"] == 1024
    assert result["best"]["aspec"]["step"] == 2
    assert result["best"]["torchspec"]["step"] == 1
    assert result["positional_proxy_steps"] == []
    assert curve[0]["aspec/weighted_distribution_overlap"] == 0.5
    assert curve[0]["ce/absolute_difference"] == 0
    with pytest.raises(ValueError, match="optimizer updates"):
        training_summary([prefix, continuation, continuation], 2)


@pytest.mark.parametrize("counter", (0, -1, 1.5, True, float("nan")))
def test_decode_requires_actual_positive_integer_verifiers_per_question(counter):
    fast, reference = decode_fixture(), decode_fixture()
    fast["runs"]["speculative"]["samples"][0][0]["meta_info"]["spec_verify_ct"] = counter
    with pytest.raises(ValueError, match="counter"):
        real_acceptance_comparison(fast, reference, bootstrap=100)


def test_training_summary_checks_validation_replicas_and_nested_finiteness(tmp_path):
    terms = {"loss": 2.0, "ce": 3.0, "l1": 1.0, "confidence": 0.8, "denominator": 10.0}
    record = {"step": 1, "learning_rate_before": 0.0006, "aspec": terms, "torchspec": terms}
    for rank in range(4):
        (tmp_path / f"steps-rank{rank}.jsonl").write_text(json.dumps(record) + "\n")
        (tmp_path / f"validation-rank{rank}.jsonl").write_text(json.dumps(record) + "\n")
    changed = copy.deepcopy(record)
    changed["torchspec"]["ce"] += 1
    (tmp_path / "validation-rank3.jsonl").write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="validation"):
        training_summary([tmp_path], 1)
    (tmp_path / "validation-rank3.jsonl").write_text(json.dumps(record) + "\n")
    record["aspec"]["training_acceptance"] = {"token_accuracy": float("nan")}
    for rank in range(4):
        (tmp_path / f"steps-rank{rank}.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        training_summary([tmp_path], 1)


def test_question_csv_uses_actual_sorted_prompt_identity_not_source_file_order(tmp_path):
    prompts = sorted(
        [[1, 2], [3, 4]], key=lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    )
    source = [
        {"data_id": "question-" + str(ids[0]), "prompt_input_ids": ids} for ids in prompts[::-1]
    ]
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in source))
    fast, reference = decode_fixture((1, 2)), decode_fixture()
    fast["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    fast["prompt_tokens_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    rows = question_comparisons(fast, reference, path)
    assert rows[0]["data_id"] == source[1]["data_id"]
    assert rows[1]["data_id"] == source[0]["data_id"]
    assert rows[0]["aspec_verifier_calls"] == 3
    assert rows[1]["torchspec_verifier_calls"] == 12
    assert rows[0]["difference"] == 2
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset hash"):
        question_comparisons(fast, reference, path)


@pytest.mark.parametrize("field", ("completion_tokens", "spec_verify_ct"))
def test_decode_rejects_fractional_or_fabricated_counters(field):
    fast, reference = decode_fixture(), decode_fixture()
    fast["runs"]["speculative"]["samples"][0][0]["meta_info"][field] = 3.5
    with pytest.raises(ValueError, match="counter"):
        real_acceptance_comparison(fast, reference, bootstrap=100)


def test_decode_checks_target_only_effective_settings_too():
    fast, reference = decode_fixture(), decode_fixture()
    fast["runs"]["target_only"]["effective_engine_args"]["disable_radix_cache"] = True
    with pytest.raises(ValueError, match="target_only"):
        real_acceptance_comparison(fast, reference, bootstrap=100)


def test_training_csv_exports_probability_acceptance_without_relabelling_accuracy(tmp_path):
    proxy = {
        "token_accuracy": 0.5,
        "per_position_accuracy": [0.5, 0.5],
        "per_position_count": [2.0, 2.0],
        "simulated_accepted_draft_tokens": 0.75,
        "per_position_accept_rate": [0.6, 0.4],
        "tau_probabilistic": 1.5,
        "expected_accepted_draft_tokens": 0.5,
        "acceptance_block_count": 2.0,
    }
    terms = {"loss": 2.0, "ce": 3.0, "l1": 1.0, "confidence": 0.8,
             "denominator": 10.0, "training_acceptance": proxy}
    record = {"step": 1, "learning_rate_before": 0.0006, "aspec": terms, "torchspec": terms}
    for rank in range(4):
        for kind in ("steps", "validation"):
            (tmp_path / f"{kind}-rank{rank}.jsonl").write_text(json.dumps(record) + "\n")
    result, curve = training_summary([tmp_path], 1)
    assert result["probabilistic_acceptance_steps"] == [1]
    assert curve[0]["aspec/tau_probabilistic"] == 1.5
    assert curve[0]["torchspec/accept_rate@0"] == 0.6
    assert curve[0]["torchspec/accuracy_0"] == 0.5
    assert curve[0]["aspec/simulated_accepted_draft_tokens"] == 0.75
