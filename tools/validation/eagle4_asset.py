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

"""Strict EAGLE4 checkpoint/native roundtrip and frozen-body audit, not serving."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from aspec.draft_artifact import load_native_draft
from tools.convert_to_hf import _extract_model_weights, _load_fsdp_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--body-asset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loaded = load_native_draft(args.asset)
    actual = loaded.state_dict()
    expected = _extract_model_weights(_load_fsdp_state_dict(args.checkpoint / "model"))
    if set(actual) != set(expected):
        raise AssertionError("Native asset tensor set differs from checkpoint")
    for key, value in expected.items():
        if value.dtype != actual[key].dtype:
            raise AssertionError("Native dtype differs: " + key)
        torch.testing.assert_close(value.cpu(), actual[key].cpu(), rtol=0, atol=0)
    result = {"all_tensors_and_dtypes_exact": True, "tensor_count": len(actual),
              "candidate_steps": int(loaded.candidate_head_steps), "policy": loaded.eagle4_policy}
    if args.body_asset:
        body = load_file(str(args.body_asset / "model.safetensors"))
        for key, value in actual.items():
            if not key.startswith("efficient_lm_head") and key != "candidate_head_steps":
                torch.testing.assert_close(value, body[key], rtol=0, atol=0)
        result["frozen_body_exact"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
