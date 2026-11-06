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

"""Bounded official-loader/greedy correctness smoke, not a serving benchmark."""

import argparse
import json
from pathlib import Path

import sglang as sgl
import torch
from transformers import AutoTokenizer

from aspec.inference.sglang_v0521 import TrainingEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", choices=("DSPARK", "DFLASH", "EAGLE3", "EAGLE"))
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument("--draft-tokens", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    if args.tp < 1 or torch.cuda.device_count() != args.tp:
        raise ValueError("Select TP idle GPUs with CUDA_VISIBLE_DEVICES")
    if not 0 < args.mem_fraction_static < 1 or (args.draft_tokens is not None and args.draft_tokens < 2):
        parser.error("memory fraction must be in (0, 1) and draft tokens >= 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "List the integers from one to five."}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    manifest = Path(args.draft) / "export_manifest.json"
    exported_algorithm = None
    if manifest.is_file():
        exported_algorithm = json.loads(manifest.read_text()).get("algorithm")
    serving_algorithm = ({"dflash2": "DFLASH", "mtp": "EAGLE"}.get(exported_algorithm)
                         or (exported_algorithm or "dspark").upper())
    algorithm = args.algorithm or serving_algorithm
    if exported_algorithm and algorithm != serving_algorithm:
        raise ValueError("Requested smoke algorithm conflicts with the exported artifact")
    if args.draft_tokens is None:
        draft_config = json.loads((Path(args.draft) / "config.json").read_text())
        args.draft_tokens = int((draft_config.get("dflash_config") or {}).get(
            "block_size", draft_config.get("block_size", 8)
        )) if algorithm == "DFLASH" else 8
    results = []
    effective_settings = []
    for draft in (None, args.draft):
        kwargs = dict(
            model_path=args.model, tp_size=args.tp, mem_fraction_static=args.mem_fraction_static,
            trust_remote_code=True, max_total_tokens=4096, max_running_requests=1,
            chunked_prefill_size=-1, log_level="warning",
            cuda_graph_config={"prefill": {"backend": "disabled"},
                               "decode": {"backend": "disabled"}},
        )
        if draft:
            kwargs.update(speculative_algorithm=algorithm, speculative_draft_model_path=draft,
                          speculative_num_draft_tokens=args.draft_tokens)
            if algorithm in {"EAGLE3", "EAGLE"}:
                kwargs.update(speculative_eagle_topk=1, speculative_num_steps=3)
        engine = sgl.Engine(**kwargs)
        try:
            effective_settings.append({
                "algorithm": engine.server_args.speculative_algorithm,
                "num_draft_tokens": engine.server_args.speculative_num_draft_tokens,
                "num_steps": engine.server_args.speculative_num_steps,
                "topk": engine.server_args.speculative_eagle_topk,
            })
            result = engine.generate(input_ids=ids, sampling_params={
                "temperature": 0, "max_new_tokens": args.tokens, "ignore_eos": True,
            })
            results.append(result)
        finally:
            TrainingEngine(engine).shutdown()
    equal = results[0]["output_ids"] == results[1]["output_ids"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sglang_path": sgl.__file__, "greedy_equal": equal, "algorithm": algorithm,
                                 "model": args.model, "draft": args.draft, "tp": args.tp,
                                 "requested_speculative_num_draft_tokens": args.draft_tokens,
                                 "effective_settings": effective_settings,
                                 "results": results}, indent=2) + "\n")
    if not equal:
        raise RuntimeError("Official speculative consumption changed greedy output")
    print("PASS official exported-draft loader and greedy output", flush=True)


if __name__ == "__main__":
    main()
