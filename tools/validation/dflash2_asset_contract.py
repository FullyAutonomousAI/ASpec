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

"""Check a published DFlash2 tensor schema without downloading its weight data."""

import argparse
import json
import struct
from pathlib import Path

import requests
import torch

from aspec.models.draft.auto import AutoDraftModelConfig
from aspec.models.draft.dflash2 import DFlash2DraftModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="z-lab/Qwen3.8-27B-DFlash2")
    parser.add_argument("--revision", default="50307d4c4cde6860d4eee73e2547cd786fe8e8a4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"
    response = requests.get(base + "/config.json", timeout=(10, 30))
    response.raise_for_status()
    raw = response.json()
    # Even if a server ignores Range, streaming closes after the header. No
    # multi-GB tensor payload is downloaded or loaded into host/device memory.
    with requests.get(base + "/model.safetensors", headers={"Range": "bytes=0-65535"},
                      stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        prefix = response.raw.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > 65528:
            raise ValueError("Header exceeds this bounded metadata request")
        header_bytes = response.raw.read(header_size)
        header = json.loads(header_bytes)
    schema = {key: value["shape"] for key, value in header.items() if key != "__metadata__"}
    with torch.device("meta"):
        model = DFlash2DraftModel(AutoDraftModelConfig.from_dict(raw))
    expected = {key: list(value.shape) for key, value in model.state_dict_for_serving().items()}
    if schema != expected:
        raise AssertionError({"missing": sorted(set(schema) - set(expected)),
                              "extra": sorted(set(expected) - set(schema)),
                              "shapes": {key: [schema[key], expected[key]] for key in schema.keys() & expected.keys()
                                         if schema[key] != expected[key]}})
    exported = model.config_for_serving()
    roundtrip = AutoDraftModelConfig.from_dict(exported)
    assert roundtrip.target_layer_ids == model.config.target_layer_ids
    assert roundtrip.target_num_hidden_layers == model.config.target_num_hidden_layers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"repo": args.repo, "revision": args.revision,
                                      "header_bytes": len(header_bytes) + 8,
                                      "tensor_count": len(schema), "schema_equal": True,
                                      "published_config": raw, "tensor_shapes": schema,
                                      "scope": "names/shapes/config only; not pretrained weight values"}, indent=2) + "\n")
    print(f"PASS {len(schema)} published DFlash2 tensor names/shapes and config roundtrip", flush=True)


if __name__ == "__main__":
    main()
