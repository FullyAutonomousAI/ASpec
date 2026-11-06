#!/usr/bin/env python3
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

"""Convert legacy pre-tokenized JSONL into ASpec's typed Parquet boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CONTRACT_KEYS = frozenset(
    {
        "data_id",
        "id",
        "input_ids",
        "loss_mask",
        "packed_loss_mask",
        "seq_len",
        "loss_tokens",
    }
)


def _pack_loss_mask(mask: list[int], *, row_number: int) -> tuple[str, int]:
    if not mask or any(value not in (0, 1, False, True) for value in mask):
        raise ValueError(f"row {row_number} loss_mask must be a non-empty list of 0/1 values")

    segments: list[int] = []
    expected = 0
    run = 0
    loss_tokens = 0
    for value in mask:
        value = int(value)
        loss_tokens += value
        if value == expected:
            run += 1
        else:
            segments.append(run)
            run = 1
            expected = value
    segments.append(run)
    return ",".join(str(value) for value in segments), loss_tokens


def _validate_packed_mask(value: Any, *, seq_len: int, row_number: int) -> tuple[str, int]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row_number} packed_loss_mask must be a non-empty string")
    try:
        segments = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"row {row_number} packed_loss_mask contains a non-integer") from exc
    if any(length < 0 for length in segments) or sum(segments) != seq_len:
        raise ValueError(f"row {row_number} packed_loss_mask length does not match input_ids")
    return value, sum(segments[1::2])


def _normalize_row(row: Any, *, row_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"row {row_number} must be a JSON object")

    input_ids = row.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(not isinstance(value, int) or isinstance(value, bool) for value in input_ids)
    ):
        raise ValueError(f"row {row_number} input_ids must be a non-empty list of integers")

    seq_len = len(input_ids)
    stored_seq_len = row.get("seq_len")
    if stored_seq_len is not None and int(stored_seq_len) != seq_len:
        raise ValueError(f"row {row_number} seq_len does not match input_ids")

    explicit_mask = row.get("loss_mask")
    packed_mask = row.get("packed_loss_mask")
    if explicit_mask is not None:
        if not isinstance(explicit_mask, list) or len(explicit_mask) != seq_len:
            raise ValueError(f"row {row_number} loss_mask length does not match input_ids")
        canonical_mask, loss_tokens = _pack_loss_mask(explicit_mask, row_number=row_number)
        if packed_mask is not None:
            stored_mask, stored_loss_tokens = _validate_packed_mask(
                packed_mask, seq_len=seq_len, row_number=row_number
            )
            if stored_mask != canonical_mask or stored_loss_tokens != loss_tokens:
                raise ValueError(f"row {row_number} packed_loss_mask disagrees with loss_mask")
    elif packed_mask is not None:
        canonical_mask, loss_tokens = _validate_packed_mask(
            packed_mask, seq_len=seq_len, row_number=row_number
        )
    else:
        raise ValueError(f"row {row_number} is missing loss_mask or packed_loss_mask")

    if loss_tokens <= 0:
        raise ValueError(f"row {row_number} has no supervised tokens")
    stored_loss_tokens = row.get("loss_tokens")
    if stored_loss_tokens is not None and int(stored_loss_tokens) != loss_tokens:
        raise ValueError(f"row {row_number} loss_tokens does not match the loss mask")

    data_id = str(row.get("data_id") or row.get("id") or f"row_{row_number:08d}")
    metadata = {key: value for key, value in row.items() if key not in _CONTRACT_KEYS}
    return {
        "data_id": data_id,
        "input_ids": input_ids,
        "packed_loss_mask": canonical_mask,
        "seq_len": seq_len,
        "loss_tokens": loss_tokens,
        "source_row": row_number,
        "source_metadata_json": json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path, *, batch_rows: int = 8) -> dict[str, Any]:
    """Convert one JSON object per line without retaining the corpus in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    if destination.suffix.lower() != ".parquet":
        raise ValueError("destination must end in .parquet")

    schema = pa.schema(
        [
            ("data_id", pa.string()),
            ("input_ids", pa.list_(pa.int64())),
            ("packed_loss_mask", pa.string()),
            ("seq_len", pa.int64()),
            ("loss_tokens", pa.int64()),
            ("source_row", pa.int64()),
            ("source_metadata_json", pa.string()),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    rows = 0
    seen_ids: set[str] = set()
    batch: list[dict[str, Any]] = []
    writer = None
    try:
        writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"row {line_number} is not valid JSON: {exc.msg}") from exc
                normalized = _normalize_row(raw, row_number=line_number)
                if normalized["data_id"] in seen_ids:
                    raise ValueError(f"duplicate data_id at row {line_number}: {normalized['data_id']}")
                seen_ids.add(normalized["data_id"])
                batch.append(normalized)
                rows += 1
                if len(batch) >= batch_rows:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
        if not rows:
            raise ValueError("source has no JSONL rows")
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        writer.close()
        writer = None
        os.replace(temporary_path, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": "aspec.pretokenized-conversion/v1",
        "source": str(source),
        "source_sha256": _sha256(source),
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "rows": rows,
        "columns": [field.name for field in schema],
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source is not a file: {args.source}")
    if args.destination.exists() and not args.overwrite:
        parser.error(f"destination already exists: {args.destination}; pass --overwrite to replace it")

    manifest = convert(args.source, args.destination, batch_rows=args.batch_rows)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
