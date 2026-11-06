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

"""Build a deterministic pre-tokenized mixed-length scheduler workload."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--lengths", default="512,768,1024,1536,2048,3072,3584,4096")
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) <= 0 or args.repetitions <= 0:
        raise ValueError("lengths and repetitions must be positive")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    requested_rows = args.repetitions * len(lengths)
    with args.source.open(encoding="utf-8") as source:
        source_rows = []
        for line in source:
            if line.strip():
                source_rows.append(json.loads(line))
            if len(source_rows) >= requested_rows:
                break
        if not source_rows:
            raise ValueError("source has no JSONL rows")
        outputs = []
        for repetition in range(args.repetitions):
            for offset, length in enumerate(lengths):
                row = source_rows[(repetition * len(lengths) + offset) % len(source_rows)]
                if len(row["input_ids"]) < length or len(row["loss_mask"]) < length:
                    raise ValueError(f"source row is shorter than requested length {length}")
                output = {
                    "id": f"mixed_{repetition:04d}_{offset:02d}_{length}",
                    "input_ids": row["input_ids"][:length],
                    "loss_mask": row["loss_mask"][:length],
                }
                outputs.append(output)

    if args.destination.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(outputs), args.destination)
    else:
        with args.destination.open("w", encoding="utf-8") as destination:
            for output in outputs:
                destination.write(json.dumps(output, separators=(",", ":")) + "\n")

    digest = hashlib.sha256(args.destination.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"rows": len(outputs), "sha256": digest, "lengths": lengths}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
