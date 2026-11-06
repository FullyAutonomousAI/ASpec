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

"""Build a deterministic fixed-length token workload for system benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    destination: Path,
    *,
    sequence_length: int,
    rows: int,
    seed: int,
    min_token_id: int,
    max_token_id: int,
    batch_rows: int,
) -> dict[str, object]:
    if sequence_length < 2 or rows < 1 or batch_rows < 1:
        raise ValueError("sequence_length >= 2, rows >= 1 and batch_rows >= 1 are required")
    if min_token_id < 0 or max_token_id <= min_token_id:
        raise ValueError("token ID range must be non-negative and non-empty")

    schema = pa.schema(
        [
            ("data_id", pa.string()),
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int8())),
            ("packed_loss_mask", pa.string()),
            ("seq_len", pa.int64()),
            ("loss_tokens", pa.int64()),
            ("source_metadata_json", pa.string()),
        ]
    )
    provenance = json.dumps(
        {
            "generator": "aspec.fixed-token-workload/v1",
            "max_token_id_exclusive": max_token_id,
            "min_token_id": min_token_id,
            "seed": seed,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    packed_mask = f"1,{sequence_length - 1}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    writer = None
    try:
        writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
        for start in range(0, rows, batch_rows):
            count = min(batch_rows, rows - start)
            ids = np.empty((count, sequence_length), dtype=np.int64)
            for offset in range(count):
                row_index = start + offset
                rng = np.random.default_rng(np.random.SeedSequence([seed, row_index]))
                ids[offset] = rng.integers(
                    min_token_id,
                    max_token_id,
                    size=sequence_length,
                    dtype=np.int64,
                )
            masks = np.ones((count, sequence_length), dtype=np.int8)
            masks[:, 0] = 0
            offsets = pa.array(
                np.arange(0, (count + 1) * sequence_length, sequence_length),
                type=pa.int32(),
            )
            table = pa.Table.from_arrays(
                [
                    pa.array(
                        [f"fixed_{sequence_length}_{start + index:08d}" for index in range(count)]
                    ),
                    pa.ListArray.from_arrays(offsets, pa.array(ids.reshape(-1))),
                    pa.ListArray.from_arrays(offsets, pa.array(masks.reshape(-1))),
                    pa.array([packed_mask] * count),
                    pa.array([sequence_length] * count, type=pa.int64()),
                    pa.array([sequence_length - 1] * count, type=pa.int64()),
                    pa.array([provenance] * count),
                ],
                schema=schema,
            )
            writer.write_table(table)
        writer.close()
        writer = None
        os.replace(temporary_path, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": "aspec.fixed-token-workload/v1",
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "sequence_length": sequence_length,
        "rows": rows,
        "seed": seed,
        "min_token_id": min_token_id,
        "max_token_id_exclusive": max_token_id,
        "supervised_tokens_per_row": sequence_length - 1,
        "prefix_cache_policy": "independent_seeded_random_tokens_per_row",
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--min-token-id", type=int, default=100)
    parser.add_argument("--max-token-id", type=int, default=30000)
    parser.add_argument("--batch-rows", type=int, default=8)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"destination already exists: {args.destination}")
    manifest = build(
        args.destination,
        sequence_length=args.sequence_length,
        rows=args.rows,
        seed=args.seed,
        min_token_id=args.min_token_id,
        max_token_id=args.max_token_id,
        batch_rows=args.batch_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
