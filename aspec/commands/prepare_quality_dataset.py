#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Prepare a deterministic Open-PerfectBlend train/eval quality split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

SOURCE_DATASET = "mlabonne/open-perfectblend"
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_conversation(messages) -> list[dict] | None:
    normalized = []
    for message in messages or []:
        if "role" in message and "content" in message:
            role = message["role"]
            content = message["content"]
        elif "from" in message and "value" in message:
            role = ROLE_MAPPING.get(message["from"], message["from"])
            content = message["value"]
        else:
            return None
        normalized.append({"role": role, "content": content})
    return normalized


def is_valid_conversation(messages: list[dict] | None) -> bool:
    if not messages or len(messages) < 2:
        return False
    assistant_chars = sum(
        len(message["content"])
        for message in messages
        if message["role"] == "assistant"
    )
    return assistant_chars >= 50


def prepare_split(
    *,
    output_dir: Path,
    train_size: int,
    eval_size: int,
    seed: int,
) -> dict:
    if train_size <= 0 or eval_size <= 0:
        raise ValueError("train_size and eval_size must be positive")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"open_perfectblend_train_{train_size}.jsonl"
    eval_path = output_dir / f"open_perfectblend_eval_{eval_size}.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (train_path, eval_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing quality artifacts: {existing}")

    dataset = load_dataset(SOURCE_DATASET)["train"]
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    eval_rows = []
    train_rows = []
    skipped = 0
    for index in indices:
        row = dataset[index]
        messages = normalize_conversation(
            row.get("conversations", row.get("conversation", []))
        )
        if not is_valid_conversation(messages):
            skipped += 1
            continue
        sample = {
            "id": f"open_perfectblend_{index}",
            "conversations": messages,
        }
        if len(eval_rows) < eval_size:
            eval_rows.append(sample)
        elif len(train_rows) < train_size:
            train_rows.append(sample)
        if len(eval_rows) == eval_size and len(train_rows) == train_size:
            break
    if len(eval_rows) != eval_size or len(train_rows) != train_size:
        raise RuntimeError(
            f"source dataset yielded only {len(train_rows)} train and {len(eval_rows)} eval rows"
        )

    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "aspec.quality-dataset/v1",
        "source_dataset": SOURCE_DATASET,
        "source_split": "train",
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "seed": seed,
        "skipped_invalid_rows": skipped,
        "train": {
            "path": str(train_path),
            "rows": len(train_rows),
            "sha256": _sha256(train_path),
        },
        "eval": {
            "path": str(eval_path),
            "rows": len(eval_rows),
            "sha256": _sha256(eval_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=65536)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_split(
                output_dir=args.output_dir,
                train_size=args.train_size,
                eval_size=args.eval_size,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
