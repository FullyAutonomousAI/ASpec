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

"""Evaluate an exported draft through official SGLang, not a training proxy."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path


def load_prompts(path, tokenizer, limit, *, enable_thinking=None):
    """Only user prefixes are evaluated; never prefill held-out assistant answers."""
    prompts = []
    generated_prompts = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if "prompt_input_ids" in row:
                ids = row["prompt_input_ids"]
            elif "input_ids" in row:
                if "loss_mask" in row or "packed_loss_mask" in row:
                    raise ValueError("training tokens require prompt_input_ids to exclude held-out answers")
                ids = row["input_ids"]
            elif isinstance(row.get("prompt"), str):
                ids = tokenizer.encode(row["prompt"])
            else:
                messages = row.get("messages", row.get("conversations"))
                if not messages:
                    raise ValueError("decode data requires input_ids, prompt or role/content messages")
                prefix = []
                for message in messages:
                    if message["role"] == "assistant":
                        break
                    prefix.append(message)
                if not prefix or prefix[-1]["role"] != "user":
                    raise ValueError("decode prefix must end with a user turn")
                options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
                ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True,
                                                     return_dict=False, **options)
            if not ids or not isinstance(ids, list) or not all(type(i) is int and i >= 0 for i in ids):
                raise ValueError("decode prompt must contain nonempty token IDs")
            if "prompt_input_ids" in row:
                # Async completion order favors short/easy answers. Select by
                # immutable prompt identity instead of journal completion time.
                key = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                generated_prompts.append((key, ids))
            else:
                prompts.append(ids)
            if len(prompts) == limit and not generated_prompts:
                break
    if generated_prompts:
        if prompts:
            raise ValueError("do not mix generated-prefix records with raw decode prompts")
        prompts = [ids for _, ids in sorted(generated_prompts)[:limit]]
    if len(prompts) != limit:
        raise ValueError(f"need {limit} held-out prompts, found {len(prompts)}")
    return prompts


def summarize(samples, *, require_verification=False):
    walls = [sum(r["wall_seconds"] for r in repeat) for repeat in samples]
    tokens = [sum(r["meta_info"]["completion_tokens"] for r in repeat) for repeat in samples]
    verifies = sum(r["meta_info"].get("spec_verify_ct", 0) for repeat in samples for r in repeat)
    if require_verification and verifies <= 0:
        raise ValueError("Speculative decode produced no actual verifier-count evidence")
    return {"repeat_wall_seconds": walls, "median_wall_seconds": statistics.median(walls),
            "repeat_tokens": tokens, "repeat_tokens_per_second": [t / w for t, w in zip(tokens, walls, strict=True)],
            "acceptance_length": sum(tokens) / verifies if verifies else None,
            "acceptance_convention": "SGLang completion_tokens / spec_verify_ct; includes target tokens",
            "verify_count": verifies}


def speculative_token_budget(config, algorithm):
    """DSpark's trained block is gamma; SGLang verifies gamma + one Target token."""
    nested = (config.get("dflash_config") or {}).get("block_size")
    flat = config.get("block_size")
    if algorithm == "DSPARK" and nested is not None and flat is not None and nested != flat:
        raise ValueError("Conflicting DSpark block_size fields")
    size = nested if nested is not None else flat
    if size is None:
        size = 7 if algorithm == "DSPARK" else 8
    if type(size) is not int or size < 2:
        raise ValueError("Exported draft block_size must be an integer >= 2")
    return size + 1 if algorithm == "DSPARK" else size


def output_checks(baseline, candidate):
    def outputs(repeat):
        return [row["output_ids"] for row in repeat]

    if (
        not baseline or not candidate or not baseline[0]
        or len(baseline) != len(candidate)
        or any(len(row) != len(baseline[0]) for row in [*baseline, *candidate])
    ):
        raise ValueError("Missing complete decode repeats")
    return {
        "greedy_equal": all(outputs(a) == outputs(b) for a, b in zip(baseline, candidate, strict=True)),
        "target_repeat_stable": all(outputs(row) == outputs(baseline[0]) for row in baseline),
        "draft_repeat_stable": all(outputs(row) == outputs(candidate[0]) for row in candidate),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Explicitly allow code supplied by the Target checkpoint")
    parser.add_argument("--disable-thinking", action="store_true", help="match a non-thinking training corpus")
    parser.add_argument("--deterministic-inference", action="store_true",
                        help="diagnose batch-invariant greedy decode; normal radix cache remains required")
    args = parser.parse_args()
    if args.output.exists() or min(args.samples, args.max_new_tokens, args.repeats, args.tp) < 1:
        raise ValueError("use positive budgets and a fresh output")
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    from aspec.inference.sglang_v0521 import TrainingEngine

    manifest = json.loads((args.draft / "export_manifest.json").read_text())
    algorithm = {"dspark": "DSPARK", "dflash": "DFLASH", "dflash2": "DFLASH",
                 "eagle3": "EAGLE3", "mtp": "EAGLE"}.get(manifest.get("algorithm"))
    if algorithm is None or manifest.get("storage_format") != "sglang":
        raise ValueError("requires an official SGLang-format draft export")
    for filename, key in (("model.safetensors", "model_sha256"), ("config.json", "config_sha256")):
        digest = hashlib.sha256()
        with (args.draft / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.get(key):
            raise ValueError("exported draft hash mismatch: " + filename)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=args.trust_remote_code)
    prompts = load_prompts(args.data, tokenizer, args.samples,
                           enable_thinking=False if args.disable_thinking else None)
    config = json.loads((args.draft / "config.json").read_text())
    draft_tokens = speculative_token_budget(config, algorithm)
    record = {"schema": "aspec.decode-evaluation/v1", "target": args.target,
              "draft": str(args.draft.resolve()), "algorithm": algorithm,
              "export_manifest_sha256": hashlib.sha256((args.draft / "export_manifest.json").read_bytes()).hexdigest(),
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "prompt_tokens_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              "settings": vars(args) | {"draft": str(args.draft), "data": str(args.data), "output": str(args.output),
                                        "temperature": 0, "warmup_requests": 1},
              "versions": {name: importlib.metadata.version(name) for name in ("sglang", "torch", "transformers")},
              "hardware": {"cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                           "devices": [{"name": torch.cuda.get_device_name(i),
                                        "capability": torch.cuda.get_device_capability(i)}
                                       for i in range(torch.cuda.device_count())]},
              "paths": {"sglang": sgl.__file__}, "runs": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    sampling = {"temperature": 0, "max_new_tokens": args.max_new_tokens}
    for name in ("target_only", "speculative"):
        kwargs = {"model_path": args.target, "tp_size": args.tp, "trust_remote_code": args.trust_remote_code}
        if args.deterministic_inference:
            kwargs["enable_deterministic_inference"] = True
        if name == "speculative":
            kwargs.update(speculative_algorithm=algorithm, speculative_draft_model_path=str(args.draft),
                          speculative_num_draft_tokens=draft_tokens)
            if algorithm in {"EAGLE", "EAGLE3"}:
                kwargs.update(speculative_eagle_topk=1, speculative_num_steps=3)
        record["runs"][name] = {"requested_engine_args": kwargs, "samples": []}
        persist()
        engine = sgl.Engine(**kwargs)
        try:
            # v0.5.21 preserves requested ServerArgs attributes separately from
            # resolution. Read the running service, not None-valued requests.
            info = engine.get_server_info()
            record["runs"][name]["server_info"] = info
            record["runs"][name]["effective_engine_args"] = {
                "algorithm": info["speculative_algorithm"],
                "num_draft_tokens": info["speculative_num_draft_tokens"],
                "num_steps": info["speculative_num_steps"],
                **{key: info.get(key) for key in
                   ("dtype", "attention_backend", "mem_fraction_static", "disable_radix_cache",
                    "enable_deterministic_inference")},
            }
            if args.deterministic_inference and (
                not info.get("enable_deterministic_inference") or info.get("disable_radix_cache")
            ):
                raise RuntimeError("Deterministic diagnostic must retain normal radix cache")
            persist()  # Expose resolved settings even while the first repeat is still running.
            engine.generate(input_ids=prompts[0], sampling_params=sampling)
            for _ in range(args.repeats):
                rows = []
                for ids in prompts:
                    start = time.perf_counter()
                    result = engine.generate(input_ids=ids, sampling_params=sampling)
                    rows.append({"output_ids": result["output_ids"], "meta_info": result["meta_info"],
                                 "wall_seconds": time.perf_counter() - start})
                record["runs"][name]["samples"].append(rows)
                persist()
        finally:
            TrainingEngine(engine).shutdown()
        record["runs"][name]["summary"] = summarize(
            record["runs"][name]["samples"], require_verification=name == "speculative",
        )
        persist()
    baseline, candidate = record["runs"]["target_only"], record["runs"]["speculative"]
    record.update(output_checks(baseline["samples"], candidate["samples"]))
    record["median_speedup"] = baseline["summary"]["median_wall_seconds"] / candidate["summary"]["median_wall_seconds"]
    record["claim"] = "bounded serial decode; not a serving throughput benchmark or convergence proof"
    persist()
    if not all(record[key] for key in ("greedy_equal", "target_repeat_stable", "draft_repeat_stable")):
        raise RuntimeError("Greedy decode equality/repeat stability failed; see recorded evidence")
    print(json.dumps({"greedy_equal": True, "speedup": record["median_speedup"],
                      "speculative": candidate["summary"]}), flush=True)


if __name__ == "__main__":
    main()
