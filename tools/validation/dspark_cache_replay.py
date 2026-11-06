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

"""Bounded cache-state diagnosis; never substitutes for the original decode gate."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.evaluation.read_text())
    if source["algorithm"] != "DSPARK" or source["settings"]["samples"] != 32:
        raise ValueError("diagnose the existing 32-prefix DSpark evaluation")
    import sglang as sgl
    from transformers import AutoTokenizer

    from aspec.inference.sglang_v0521 import TrainingEngine
    from tools.evaluate_draft import load_prompts
    from tools.prepare_target_dataset import sha256

    for name, version in source["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("evaluation environment changed: " + name)
    draft = Path(source["draft"])
    manifest = json.loads((draft / "export_manifest.json").read_text())
    if sha256(draft / "export_manifest.json") != source["export_manifest_sha256"]:
        raise ValueError("export manifest changed")
    for name, key in (("model.safetensors", "model_sha256"), ("config.json", "config_sha256")):
        if sha256(draft / name) != manifest[key]:
            raise ValueError("export asset changed: " + name)
    if sha256(source["settings"]["data"]) != source["data_sha256"]:
        raise ValueError("held-out data changed")
    tokenizer = AutoTokenizer.from_pretrained(source["target"], trust_remote_code=True)
    prompts = load_prompts(source["settings"]["data"], tokenizer, 32)
    if hashlib.sha256(json.dumps(prompts).encode()).hexdigest() != source["prompt_tokens_sha256"]:
        raise ValueError("prompt selection changed")
    prompts = prompts[:8]  # Fixed prefix of the original selection, not chosen for matching outputs.
    sampling = {"temperature": 0, "max_new_tokens": source["settings"]["max_new_tokens"]}
    record = {"schema": "aspec.dspark-cache-replay/v1", "diagnostic_only": True,
              "original_evaluation": str(args.evaluation), "original_sha256": sha256(args.evaluation),
              "probe_sha256": sha256(__file__), "original_sample_indices": list(range(8)),
              "sampling": sampling, "runs": {}, "performance_claim": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    for name in ("target_only", "speculative"):
        kwargs = dict(source["runs"][name]["requested_engine_args"])
        record["runs"][name] = {"engine_args": kwargs, "cases": {}}
        persist()
        engine = sgl.Engine(**kwargs)
        try:
            engine.generate(input_ids=prompts[0], sampling_params=sampling)
            for case in ("first_pass", "cached_replay", "flushed_replay"):
                if case == "flushed_replay":
                    result = engine.flush_cache()
                    if not result.success:
                        raise RuntimeError("own-engine cache flush failed")
                    record["runs"][name]["flush_success"] = True
                    # Restore the original initial condition: exactly one warmup request.
                    engine.generate(input_ids=prompts[0], sampling_params=sampling)
                rows = []
                for ids in prompts:
                    result = engine.generate(input_ids=ids, sampling_params=sampling)
                    if name == "speculative" and result["meta_info"].get("spec_verify_ct", 0) <= 0:
                        raise RuntimeError("diagnosis did not exercise the real verifier")
                    rows.append({"output_ids": result["output_ids"], "meta_info": result["meta_info"]})
                record["runs"][name]["cases"][case] = rows
                persist()
                print(f"CACHE_CASE {name} {case}: {len(rows)} complete", flush=True)
        finally:
            TrainingEngine(engine).shutdown()

    def equal(a, b):
        return sum(x["output_ids"] == y["output_ids"] for x, y in zip(a, b, strict=True))

    a, b = [record["runs"][name]["cases"] for name in ("target_only", "speculative")]
    record["cross_engine_equal"] = {case: equal(a[case], b[case]) for case in a}
    record["within_engine_equal_to_first"] = {
        name: {case: equal(run["cases"]["first_pass"], rows) for case, rows in run["cases"].items()}
        for name, run in record["runs"].items()
    }
    record["original_greedy_equal"] = source["greedy_equal"]
    record["original_evaluation_replaced"] = False
    persist()
    print(json.dumps({key: record[key] for key in
                      ("cross_engine_equal", "within_engine_equal_to_first", "original_greedy_equal")}))


if __name__ == "__main__":
    main()
