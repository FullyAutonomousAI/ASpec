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

"""Regenerate disjoint single-turn corpora with the training Target via SGLang.

Uses DeepSpec's public non-thinking generation recipe, not its offline feature
cache. Exact generated IDs feed ASpec's existing pretokenized input contract.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_prompts(paths, tokenizer, *, max_length, min_answer_tokens):
    """Eval has first claim on duplicate questions; never use source answers."""
    selected, seen, stats = {}, set(), {}
    for split in ("eval", "train"):
        rows, counts = [], {"source_rows": 0, "duplicate_questions": 0, "too_long": 0}
        with Path(paths[split]).open() as stream:
            for line in stream:
                source = json.loads(line)
                counts["source_rows"] += 1
                messages = source.get("messages", source.get("conversations", []))
                prefix = []
                for message in messages:
                    if message["role"] == "assistant":
                        break
                    if message["role"] not in {"system", "user"} or not isinstance(message["content"], str):
                        raise ValueError("single-turn recipe requires role/content text messages")
                    prefix.append(message)
                if [m["role"] for m in prefix] not in (["user"], ["system", "user"]):
                    raise ValueError("single-turn recipe needs an optional system then one user turn")
                # Sharing the same question with a different system prompt is
                # still leakage for this recipe's held-out split.
                question = " ".join(prefix[-1]["content"].split())
                if not question:
                    raise ValueError("empty user question")
                key = hashlib.sha256(question.encode()).hexdigest()
                if key in seen:
                    counts["duplicate_questions"] += 1
                    continue
                seen.add(key)
                ids = tokenizer.apply_chat_template(
                    prefix, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=False,
                )
                if len(ids) + min_answer_tokens > max_length:
                    counts["too_long"] += 1
                    continue
                rows.append({"id": key, "source_id": str(source.get("id", "")),
                             "input_ids": ids})
        selected[split], stats[split] = rows, counts
        if not rows:
            raise ValueError(f"no eligible {split} prompts")
    return selected, stats


def collect_resumed(path, rows):
    """A partial final line is not silently swallowed: fix/recover it explicitly."""
    completed = set()
    expected = {row["id"]: row["input_ids"] for row in rows}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row["id"]
                if key not in expected or key in completed or row["prompt_input_ids"] != expected[key]:
                    raise ValueError("generated journal contains unknown, duplicate or changed prompts")
                validate_generated(row)
                completed.add(key)
    return completed


def validate_generated(row):
    prompt, tokens = row["prompt_input_ids"], row["input_ids"]
    if (not prompt or tokens[:len(prompt)] != prompt or len(tokens) <= len(prompt)
            or any(type(i) is not int or i < 0 for i in tokens)
            or row["loss_mask"] != [0] * len(prompt) + [1] * (len(tokens) - len(prompt))):
        raise ValueError("generated row has invalid tokens or supervision boundary")


async def regenerate(engine, rows, path, *, batch_size, max_length, seed):
    done = collect_resumed(path, rows)
    pending = [row for row in rows if row["id"] not in done]
    iterator, completed = iter(pending), len(done)
    with path.open("a", buffering=1) as stream:
        async def worker():
            nonlocal completed
            # Refill each completed request immediately: a 4K answer must not
            # strand the other GPUs behind a fixed synchronous batch barrier.
            for source in iterator:
                settings = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                            "max_new_tokens": max_length - len(source["input_ids"]),
                            "sampling_seed": (seed + int(source["id"][:8], 16)) % (2**31)}
                output = await engine.async_generate(input_ids=source["input_ids"],
                                                     sampling_params=settings)
                generated = output["output_ids"]
                reason = output["meta_info"]["finish_reason"]
                if reason.get("type") not in {"stop", "length"} or not generated:
                    raise RuntimeError(f"Target generation failed: {reason}")
                row = {"id": source["id"], "source_id": source["source_id"],
                       "prompt_input_ids": source["input_ids"],
                       "input_ids": source["input_ids"] + generated,
                       "loss_mask": [0] * len(source["input_ids"]) + [1] * len(generated),
                       "finish_reason": reason["type"]}
                validate_generated(row)
                if len(row["input_ids"]) > max_length:
                    raise RuntimeError("Target exceeded the configured token budget")
                stream.write(json.dumps(row) + "\n")
                completed += 1
                if completed % batch_size == 0 or completed == len(rows):
                    print(f"{path.stem}: {completed}/{len(rows)}", flush=True)

        tasks = [asyncio.create_task(worker()) for _ in range(min(batch_size, len(pending)))]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="local immutable Target snapshot")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128, help="maximum concurrent requests; refilled continuously")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-answer-tokens", type=int, default=32, help="reserved generation room, not forced output length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.dp, args.batch_size, args.min_answer_tokens) < 1 or args.max_length <= args.min_answer_tokens:
        parser.error("positive budgets and max-length > min-answer-tokens required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("nonempty generation directory; use --resume for the same recipe")
    from transformers import AutoTokenizer

    target_files = sorted({*args.target.glob("*.json"), *args.target.glob("*.jinja"),
                           *args.target.glob("*.safetensors")})
    if not any(path.suffix == ".safetensors" for path in target_files):
        raise ValueError("requires a local safetensors Target snapshot")
    binding = {"schema": "aspec.target-generated-data/v1",
               "sources": {key: sha256(path) for key, path in
                           (("train", args.train_data), ("eval", args.eval_data))},
               "target": {path.name: sha256(path) for path in target_files},
               "settings": {key: getattr(args, key) for key in
                            ("dp", "batch_size", "max_length", "min_answer_tokens", "seed")},
               "generation": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                              "enable_thinking": False, "turns": "first_user_only"},
               "versions": {name: importlib.metadata.version(name) for name in ("sglang", "transformers", "torch")},
               "tool_sha256": sha256(__file__)}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["binding"] != binding:
            raise ValueError("resume input/model/settings identity changed")
    elif any(args.output_dir.iterdir()):
        raise ValueError("cannot resume generation without its identity manifest")
    else:
        manifest = {"binding": binding, "status": "generating"}
        write_json(manifest_path, manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    selected, stats = select_prompts({"train": args.train_data, "eval": args.eval_data}, tokenizer,
                                     max_length=args.max_length, min_answer_tokens=args.min_answer_tokens)
    paths = {split: args.output_dir / f"{split}.jsonl" for split in selected}
    for split, path in paths.items():
        collect_resumed(path, selected[split])
    if manifest["status"] == "complete":
        for split, path in paths.items():
            if sha256(path) != manifest["outputs"][split]["sha256"]:
                raise ValueError("completed corpus was modified")
        print(json.dumps(manifest, indent=2))
        return
    import sglang as sgl

    from aspec.inference.sglang_v0521 import TrainingEngine

    engine = sgl.Engine(model_path=str(args.target), tp_size=1, dp_size=args.dp,
                        context_length=args.max_length, random_seed=args.seed)

    async def run_splits():
        for split, path in paths.items():
            await regenerate(engine, selected[split], path, batch_size=args.batch_size,
                             max_length=args.max_length, seed=args.seed)

    try:
        asyncio.run(run_splits())
    finally:
        TrainingEngine(engine).shutdown()
    manifest.update(status="complete", selection=stats,
                    outputs={split: {"rows": len(rows), "sha256": sha256(paths[split])}
                             for split, rows in selected.items()})
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
