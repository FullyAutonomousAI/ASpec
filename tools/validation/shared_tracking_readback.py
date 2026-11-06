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

"""Read actual TensorBoard/W&B offline records from a shared-Target smoke run.

Diagnostic only, not a user training entry point. The read-only W&B framing
parser is reimplemented from its published format (not imported SDK internals):
https://github.com/wandb/wandb/blob/v0.19.10/wandb/sdk/internal/datastore.py
"""

import argparse
import json
import math
import struct
import zlib
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from wandb.proto.wandb_internal_pb2 import Record

from aspec.reporting import build_run_report


def records(path):
    raw = path.read_bytes()
    assert struct.unpack_from("<4sHB", raw) == (b":W&B", 0xBEE1, 0), "unknown W&B file format"
    offset, partial = 7, bytearray()
    while offset < len(raw):
        remaining = 32768 - offset % 32768
        if remaining < 7:
            assert not any(raw[offset:offset + remaining]), "invalid block padding"
            offset += remaining
            continue
        checksum, size, kind = struct.unpack_from("<IHB", raw, offset)
        payload = raw[offset + 7:offset + 7 + size]
        assert kind in (1, 2, 3, 4) and size + 7 <= remaining and len(payload) == size
        assert zlib.crc32(bytes([kind]) + payload) & 0xFFFFFFFF == checksum
        offset += 7 + size
        if kind in (1, 2):
            assert not partial, "unfinished record"
        else:
            assert partial, "missing record start"
        partial.extend(payload)
        if kind in (1, 4):
            yield Record.FromString(bytes(partial))
            partial.clear()
    assert not partial, "truncated record"


def read_experiment(directory):
    report = build_run_report(directory)
    tracking = report["tracking"]
    assert tracking["state"] == "closed" and tracking["tensorboard"]["enabled"]
    assert tracking["wandb"]["mode"] == "offline"
    events = EventAccumulator(tracking["tensorboard"]["log_dir"], size_guidance={"scalars": 0}).Reload()
    wandb_file, = Path(tracking["wandb"]["directory"]).parent.glob("run-*.wandb")
    histories, identities, exits = [], [], []
    for record in records(wandb_file):
        if record.HasField("history"):
            histories.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                              for item in record.history.item})
        if record.HasField("run"):
            identities.append(record.run.run_id)
        if record.HasField("exit"):
            exits.append(record.exit.exit_code)
    assert identities and set(identities) == {tracking["wandb"]["run_id"]}
    assert exits == [tracking["exit_code"]]
    result = {"state": report["state"], "health": report["health"],
              "wandb_run_id": identities[0], "wandb_group": tracking["wandb"]["group"],
              "exit_code": tracking["exit_code"], "resume_source": tracking["resume_source"],
              "training": report["training"], "record_files": [str(wandb_file)], "curves": {}}
    for key, axis in (("train/avg_loss", "train/step"), ("eval/avg_loss", "eval/step")):
        tb_curve = [(event.step, event.value) for event in events.Scalars(key)] if key in events.Tags()["scalars"] else []
        wb_curve = [(entry[axis], entry[key]) for entry in histories if key in entry]
        assert len(tb_curve) == len(wb_curve), (directory.name, key, tb_curve, wb_curve)
        assert all(a[0] == b[0] and math.isclose(a[1], b[1], rel_tol=1e-6, abs_tol=1e-7)
                   for a, b in zip(tb_curve, wb_curve)), (directory.name, key, tb_curve, wb_curve)
        result["curves"][key] = {"steps": [step for step, _ in tb_curve],
                                 "values": [value for _, value in wb_curve], "backends_agree": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    children = {path.parent.name: read_experiment(path.parent)
                for path in sorted(args.output.glob("*/tracking.json"))}
    assert children, "no per-Draft tracking metadata"
    assert len({child["wandb_run_id"] for child in children.values()}) == len(children)
    assert len({child["wandb_group"] for child in children.values()}) == 1
    print(json.dumps({"output": str(args.output), "experiments": children}, indent=2))


if __name__ == "__main__":
    main()
