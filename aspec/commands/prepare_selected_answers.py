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

"""Retokenize original answers for the IDs in an existing frozen corpus.

Reuses the public parser and typed-Parquet converter. Selection token IDs are
never interpreted with a different tokenizer. This prepares workflow data,
not Target-generated labels or a quality-equivalent corpus.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_sources(selection):
    import pyarrow.parquet as pq

    table = pq.read_table(selection, columns=["data_id", "source_metadata_json"])
    selected = []
    for row in table.to_pylist():
        identity = row["data_id"]
        if identity is None or not str(identity):
            raise ValueError("Selection must preserve a nonempty data_id")
        metadata = json.loads(row["source_metadata_json"])
        source_id = metadata.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Selection must preserve a nonempty source_id")
        selected.append((str(identity), source_id))
    for column in (0, 1):
        if len({row[column] for row in selected}) != len(selected):
            raise ValueError("Selection contains duplicate document or source identities")
    if not selected:
        raise ValueError("Empty selection")
    return selected


def original_answers(source, selected):
    """Join by source ID and retain the same first-user/first-answer boundary."""
    from aspec.commands.prepare_quality_dataset import normalize_conversation

    required = {source_id for _, source_id in selected}
    found = {}
    with Path(source).open() as stream:
        for line in stream:
            row = json.loads(line)
            source_id = str(row.get("id", ""))
            if source_id not in required:
                continue
            if source_id in found:
                raise ValueError(f"Duplicate original source ID: {source_id}")
            messages = normalize_conversation(row.get("conversations"))
            offset = int(bool(messages and messages[0]["role"] == "system"))
            if (
                not messages
                or len(messages) < offset + 2
                or messages[offset]["role"] != "user"
                or messages[offset + 1]["role"] != "assistant"
                or any(not isinstance(m["content"], str) for m in messages[: offset + 2])
            ):
                raise ValueError(f"Invalid first question/answer for {source_id}")
            found[source_id] = messages[: offset + 2]
    missing = required - found.keys()
    if missing:
        raise ValueError(f"Original source is missing {len(missing)} selected IDs")
    return [(identity, source_id, found[source_id]) for identity, source_id in selected]


def prepare(*, selection, source, target, template, output, max_length=4096, min_loss_tokens=16,
            trust_remote_code=False):
    from transformers import AutoTokenizer

    from aspec.commands.convert_pretokenized_jsonl import convert
    from aspec.data.parse import create_parser
    from aspec.data.template import TEMPLATE_REGISTRY

    if max_length < 1 or min_loss_tokens < 1:
        raise ValueError("Length and supervision budgets must be positive")
    selected = selected_sources(selection)
    rows = original_answers(source, selected)
    tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=trust_remote_code)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get(template))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "documents.jsonl"
    counts = {"selected": len(selected), "over_length": 0, "short_supervision": 0, "admitted": 0}
    lengths, supervision = [], []
    with path.open("w") as stream:
        for identity, source_id, messages in rows:
            formatted = parser.format(messages, add_generation_prompt=False, enable_thinking=False)
            ids, mask = parser.parse(
                formatted, max_length + 1, preformatted=True, last_turn_only=False
            )
            if ids.numel() > max_length:
                counts["over_length"] += 1
                continue
            if int(mask.sum()) < min_loss_tokens:
                counts["short_supervision"] += 1
                continue
            stream.write(
                json.dumps(
                    {
                        "data_id": identity,
                        "source_id": source_id,
                        "input_ids": ids.tolist(),
                        "loss_mask": mask.tolist(),
                    }
                )
                + "\n"
            )
            counts["admitted"] += 1
            lengths.append(ids.numel())
            supervision.append(int(mask.sum()))
    if not lengths:
        raise ValueError("No original answers passed admission")
    conversion = convert(path, output / "documents.parquet", batch_rows=256)
    target = Path(target)
    manifest = {
        "schema": "aspec.selected-original-answers/v1",
        "answer_regeneration": False,
        "answer_source": "original first assistant turn",
        "quality_claim": False,
        "selection": str(selection),
        "selection_sha256": sha256(selection),
        "source": str(source),
        "source_sha256": sha256(source),
        "selection_order_sha256": hashlib.sha256(
            "\n".join(identity for identity, _ in selected).encode()
        ).hexdigest(),
        "target": str(target),
        "template": template,
        "enable_thinking": False,
        "trust_remote_code": trust_remote_code,
        "tokenizer_files": {
            p.name: sha256(p)
            for p in sorted(target.iterdir())
            if p.name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            }
        },
        "max_length": max_length,
        "min_loss_tokens": min_loss_tokens,
        "counts": counts,
        "input_tokens": sum(lengths),
        "supervised_positions": sum(supervision),
        "lengths": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "conversion": conversion,
        "tool_sha256": sha256(__file__),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "source", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-loss-tokens", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Explicitly allow code supplied by the Target checkpoint")
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
