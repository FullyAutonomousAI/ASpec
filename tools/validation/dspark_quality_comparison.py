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

"""Summarize matched DSpark training and paired real-verifier acceptance evidence."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError("Nonfinite training evidence")


def check_supervision(row):
    require_finite(row)
    fast, reference = row["aspec"], row["torchspec"]
    if fast["denominator"] <= 0 or fast["denominator"] != reference["denominator"]:
        raise ValueError("Matched training supervision differs")
    proxies = [group.get("training_acceptance") for group in (fast, reference)]
    if (proxies[0] is None) != (proxies[1] is None):
        raise ValueError("Matched positional proxy coverage differs")
    if proxies[0] is not None and (
        proxies[0]["per_position_count"] != proxies[1]["per_position_count"]
    ):
        raise ValueError("Matched positional supervision differs")
    if proxies[0] is not None:
        observed = ["tau_probabilistic" in proxy for proxy in proxies]
        if observed[0] != observed[1]:
            raise ValueError("Matched probability acceptance coverage differs")
        if observed[0] and proxies[0]["acceptance_block_count"] != proxies[1]["acceptance_block_count"]:
            raise ValueError("Matched probability acceptance block counts differ")


def training_summary(runs, expected_steps):
    rows, validations = [], []
    for directory in map(Path, runs):
        ranks = [read_rows(directory / f"steps-rank{rank}.jsonl") for rank in range(4)]
        if any(len(rank) != len(ranks[0]) for rank in ranks):
            raise ValueError("Training rank frontiers differ")
        for index, row in enumerate(ranks[0]):
            for peer in ranks[1:]:
                if any(
                    peer[index].get(key) != row.get(key)
                    for key in ("step", "learning_rate_before", "aspec", "torchspec")
                ):
                    raise ValueError("Native DP replicas disagree")
            check_supervision(row)
            rows.append(row)
        validation_ranks = [
            read_rows(directory / f"validation-rank{rank}.jsonl") for rank in range(4)
        ]
        if any(peer != validation_ranks[0] for peer in validation_ranks[1:]):
            raise ValueError("Native DP validation replicas disagree")
        for row in validation_ranks[0]:
            check_supervision(row)
            validations.append({"run": str(directory), **row})
    if [row["step"] for row in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("Missing, repeated or out-of-order optimizer updates")
    if not validations or max(row["step"] for row in validations) != expected_steps:
        raise ValueError("Missing final held-out validation")
    validation_steps = [row["step"] for row in validations]
    if validation_steps != sorted(set(validation_steps)) or min(validation_steps) < 1:
        raise ValueError("Repeated or out-of-order held-out validation")
    final = [row for row in validations if row["step"] == expected_steps][-1]
    best = {
        name: min(validations, key=lambda row: row[name]["ce"])
        for name in ("aspec", "torchspec")
    }
    denominator = sum(row["aspec"]["denominator"] for row in rows)
    result = {
        "steps": expected_steps,
        "documents": expected_steps * 512,
        "selection": "minimum pooled validation CE; acceptance questions never select checkpoints",
        "best": {
            name: {"run": row["run"], "step": row["step"], "metrics": row[name]}
            for name, row in best.items()
        },
        "final_validation": final,
        "pooled_training": {
            name: {
                key: sum(row[name][key] * row[name]["denominator"] for row in rows) / denominator
                for key in ("loss", "ce", "l1", "confidence")
            }
            for name in ("aspec", "torchspec")
        },
        "positional_proxy_steps": [
            row["step"] for row in rows if "training_acceptance" in row["torchspec"]
        ],
        "probabilistic_acceptance_steps": [
            row["step"] for row in rows
            if "tau_probabilistic" in row["torchspec"].get("training_acceptance", {})
        ],
        "proxy_warning": "teacher-forced overlap/marginal accuracy is not real decode acceptance",
    }
    curve = []
    for row in rows:
        record = {"step": row["step"], "learning_rate": row.get("learning_rate_before")}
        for name in ("aspec", "torchspec"):
            group = row[name]
            record.update(
                {
                    f"{name}/{key}": group[key]
                    for key in ("loss", "ce", "l1", "confidence", "denominator")
                }
            )
            record[f"{name}/weighted_distribution_overlap"] = 1 - 0.5 * group["l1"]
            proxy = group.get("training_acceptance")
            if proxy:
                record[f"{name}/simulated_accepted_draft_tokens"] = proxy[
                    "simulated_accepted_draft_tokens"
                ]
                record[f"{name}/token_accuracy"] = proxy["token_accuracy"]
                for index, accuracy in enumerate(proxy["per_position_accuracy"]):
                    record[f"{name}/accuracy_{index}"] = accuracy
                if "tau_probabilistic" in proxy:
                    record[f"{name}/tau_probabilistic"] = proxy["tau_probabilistic"]
                    record[f"{name}/expected_accepted_draft_tokens"] = proxy[
                        "expected_accepted_draft_tokens"
                    ]
                    record[f"{name}/acceptance_block_count"] = proxy["acceptance_block_count"]
                    for index, rate in enumerate(proxy["per_position_accept_rate"]):
                        record[f"{name}/accept_rate@{index}"] = rate
        for key in ("loss", "ce", "l1", "confidence"):
            actual, reference = row["aspec"][key], row["torchspec"][key]
            record[f"{key}/signed_difference"] = actual - reference
            record[f"{key}/absolute_difference"] = abs(actual - reference)
            record[f"{key}/relative_difference"] = (
                (actual - reference) / abs(reference) if reference else None
            )
        curve.append(record)
    return result, curve


def real_acceptance_comparison(fast, reference, *, bootstrap=10000):
    import numpy as np

    for key in ("algorithm", "target", "prompt_tokens_sha256", "data_sha256", "versions"):
        if fast[key] != reference[key]:
            raise ValueError(f"Decode protocol differs: {key}")
    for key in (
        "samples",
        "repeats",
        "max_new_tokens",
        "tp",
        "temperature",
        "disable_thinking",
        "deterministic_inference",
    ):
        if fast["settings"].get(key) != reference["settings"].get(key):
            raise ValueError(f"Decode settings differ: {key}")
    for key in (
        "algorithm",
        "num_draft_tokens",
        "num_steps",
        "dtype",
        "attention_backend",
        "mem_fraction_static",
        "disable_radix_cache",
        "enable_deterministic_inference",
    ):
        for mode in ("target_only", "speculative"):
            if fast["runs"][mode]["effective_engine_args"].get(key) != reference["runs"][mode][
                "effective_engine_args"
            ].get(key):
                raise ValueError(f"Effective inference differs: {mode}.{key}")
    size, repeats = fast["settings"]["samples"], fast["settings"]["repeats"]
    arrays, repeat_rates = {}, {}
    oracle = fast["runs"]["target_only"]["samples"][0]
    if len(oracle) != size:
        raise ValueError("Incomplete target oracle")
    for name, artifact in (("aspec", fast), ("torchspec", reference)):
        for mode in ("target_only", "speculative"):
            observed = artifact["runs"][mode]["samples"]
            if len(observed) != repeats or any(len(row) != size for row in observed):
                raise ValueError("Incomplete decode repeats")
            if any(
                row["output_ids"] != oracle[index]["output_ids"]
                for repeat in observed
                for index, row in enumerate(repeat)
            ):
                raise ValueError("Common greedy outputs or repeat stability differ")
        spec = artifact["runs"]["speculative"]["samples"]
        for repeat in spec:
            for row in repeat:
                tokens = row["meta_info"]["completion_tokens"]
                verifies = row["meta_info"].get("spec_verify_ct")
                if (
                    type(tokens) is not int
                    or tokens != len(row["output_ids"])
                    or type(verifies) is not int
                    or verifies <= 0
                ):
                    raise ValueError("Invalid actual verifier counters")
        values = np.array(
            [
                [
                    [
                        row["meta_info"]["completion_tokens"],
                        row["meta_info"].get("spec_verify_ct", 0),
                    ]
                    for row in repeat
                ]
                for repeat in spec
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or (values < 0).any() or values[:, :, 1].sum() <= 0:
            raise ValueError("Invalid actual verifier counters")
        arrays[name] = values.sum(axis=0)
        repeat_rates[name] = (values[:, :, 0].sum(axis=1) / values[:, :, 1].sum(axis=1)).tolist()
    actual, base = arrays["aspec"], arrays["torchspec"]
    if not np.array_equal(actual[:, 0], base[:, 0]):
        raise ValueError("Common outputs have inconsistent completion-token counters")
    # Resample questions, not repeats: three repeats are not 3*N independent examples.
    indices = np.random.default_rng(42).integers(0, size, size=(bootstrap, size))
    totals = {name: values[indices].sum(axis=1) for name, values in arrays.items()}
    if any((value[:, 1] <= 0).any() for value in totals.values()):
        raise ValueError("Bootstrap includes an empty verification denominator")
    rates = {name: value[:, 0] / value[:, 1] for name, value in totals.items()}
    point = {name: float(value[:, 0].sum() / value[:, 1].sum()) for name, value in arrays.items()}
    return {
        "questions": size,
        "repeats": repeats,
        "common_greedy_outputs": True,
        "acceptance_length": point,
        "repeat_acceptance_length": repeat_rates,
        "difference": point["aspec"] - point["torchspec"],
        "relative_difference": point["aspec"] / point["torchspec"] - 1,
        "paired_question_bootstrap_95ci": {
            "difference": np.quantile(
                rates["aspec"] - rates["torchspec"], [0.025, 0.975]
            ).tolist(),
            "relative_difference": np.quantile(
                rates["aspec"] / rates["torchspec"] - 1, [0.025, 0.975]
            ).tolist(),
        },
        "actual_counters": {
            name: {
                "completion_tokens": int(value[:, 0].sum()),
                "verifier_calls": int(value[:, 1].sum()),
            }
            for name, value in arrays.items()
        },
        "convention": "completion tokens / actual verifier calls, including Target tokens",
        "inference": fast["runs"]["speculative"]["effective_engine_args"],
        "interpretation": "CI across questions; containing zero does not prove equivalence",
    }


def question_comparisons(fast, reference, data=None):
    """Rows follow the evaluator's actual prompt order, never the source-file order."""
    labels = None
    size = fast["settings"]["samples"]
    if data is not None:
        from tools.evaluate_draft import load_prompts

        if hashlib.sha256(Path(data).read_bytes()).hexdigest() != fast["data_sha256"]:
            raise ValueError("Question identity dataset hash differs")
        prompts = load_prompts(data, None, size)
        if hashlib.sha256(json.dumps(prompts).encode()).hexdigest() != fast["prompt_tokens_sha256"]:
            raise ValueError("Question identity prompt order differs")
        source_rows = read_rows(data)
        by_tokens = {tuple(row["prompt_input_ids"]): row["data_id"] for row in source_rows}
        if len(by_tokens) != len(source_rows):
            raise ValueError("Duplicate question identity")
        labels = [by_tokens[tuple(prompt)] for prompt in prompts]
    rows = []
    for index in range(size):
        row = {"question_index": index}
        if labels is not None:
            row["data_id"] = labels[index]
        for name, artifact in (("aspec", fast), ("torchspec", reference)):
            samples = [
                repeat[index]["meta_info"] for repeat in artifact["runs"]["speculative"]["samples"]
            ]
            tokens = sum(value["completion_tokens"] for value in samples)
            verifies = sum(value["spec_verify_ct"] for value in samples)
            row[name + "_completion_tokens"] = tokens
            row[name + "_verifier_calls"] = verifies
            row[name + "_acceptance_length"] = tokens / verifies
        row["difference"] = row["aspec_acceptance_length"] - row["torchspec_acceptance_length"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--runs", nargs="+", required=True)
    train.add_argument("--steps", type=int, default=1180)
    train.add_argument("--output", type=Path, required=True)
    decode = sub.add_parser("decode")
    for name in ("fast", "reference", "output"):
        decode.add_argument("--" + name, required=True, type=Path)
    decode.add_argument(
        "--data", type=Path, help="Saved prompt_input_ids/data_id JSONL for verified CSV labels"
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise ValueError("Use a fresh report output")
    if args.command == "train":
        result, curve = training_summary(args.runs, args.steps)
        with args.output.with_suffix(".csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=sorted({key for row in curve for key in row})
            )
            writer.writeheader()
            writer.writerows(curve)
    else:
        fast, reference = json.loads(args.fast.read_text()), json.loads(args.reference.read_text())
        result = real_acceptance_comparison(fast, reference)
        curve = question_comparisons(fast, reference, args.data)
        with args.output.with_suffix(".csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
