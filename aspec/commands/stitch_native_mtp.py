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

"""Combine a native MTP export with a local verifier without duplicate MTP keys."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from aspec.draft_artifact import _sha256
from aspec.models.draft.base import load_tensor_from_pretrained


def stitch_native_mtp(*, target, draft, output):
    target, draft, output = (Path(path).expanduser().resolve() for path in (target, draft, output))
    if output.exists():
        raise FileExistsError(output)
    if not target.is_dir() or not draft.is_dir():
        raise ValueError("Stitching requires local verifier and exported draft directories")
    manifest = json.loads((draft / "export_manifest.json").read_text())
    if manifest.get("algorithm") != "mtp" or manifest.get("storage_format", "sglang") != "sglang":
        raise ValueError("Stitching requires a native MTP serving export")
    for filename, field in (("config.json", "config_sha256"), ("model.safetensors", "model_sha256")):
        if _sha256(draft / filename) != manifest.get(field):
            raise ValueError("MTP export checksum mismatch: " + filename)
    raw = json.loads((target / "config.json").read_text())
    text = raw.get("text_config", raw)
    exported = json.loads((draft / "config.json").read_text())
    if text.get("model_type") != "qwen3_5_text" or any(
        text.get(key) != exported.get(key) for key in (
            "hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps",
            "rope_parameters", "attention_bias", "tie_word_embeddings",
        )
    ):
        raise ValueError("MTP export semantics disagree with the native verifier")
    weights = load_file(str(draft / "model.safetensors"))
    embedding_key = "model.language_model.embed_tokens.weight" if "text_config" in raw else "model.embed_tokens.weight"
    for name, key in (("model.embed_tokens.weight", embedding_key), ("lm_head.weight", "lm_head.weight")):
        expected = load_tensor_from_pretrained(str(target), key)
        value = weights.get(name)
        if value is None or value.shape != expected.shape or not torch.equal(value, expected.to(value.dtype)):
            raise ValueError("MTP shared embedding/head does not match the verifier: " + name)
    mtp = {key: value for key, value in weights.items() if key.startswith("mtp.")}
    if not mtp or set(weights) != set(mtp) | {"model.embed_tokens.weight", "lm_head.weight"}:
        raise ValueError("MTP export has missing or unexpected tensors")
    del weights

    index_file = target / "model.safetensors.index.json"
    if index_file.is_file():
        source_map = json.loads(index_file.read_text())["weight_map"]
    else:
        with safe_open(target / "model.safetensors", framework="pt") as handle:
            source_map = {name: "model.safetensors" for name in handle.keys()}
    original_mtp = {name for name in source_map if name.startswith("mtp.")}
    if set(mtp) != original_mtp:
        raise ValueError("Native verifier and export contain different MTP tensor sets")
    for name, value in mtp.items():
        shape = load_tensor_from_pretrained(str(target), name).shape
        if value.shape != shape:
            raise ValueError("MTP tensor shape mismatch: " + name)
    shard_names = set(source_map.values())
    if any(Path(name).name != name or Path(name).suffix != ".safetensors" for name in shard_names):
        raise ValueError("Verifier index must reference flat local safetensors shards")

    mtp_name = "aspec-native-mtp.safetensors"
    if mtp_name in shard_names:
        raise ValueError("Verifier already uses the reserved stitched MTP shard name")
    # Preflight above is read-only. A new directory is intentionally never
    # overwritten; failed outputs are preserved for diagnosis, not silently reused.
    output.mkdir(parents=True, exist_ok=False)
    for entry in target.iterdir():
        if entry.is_file() and entry.name not in shard_names and entry.suffix != ".safetensors" and entry.name != index_file.name:
            shutil.copy2(entry, output / entry.name)
    result_map, total_size = {}, 0
    for shard in sorted(shard_names):
        with safe_open(target / shard, framework="pt") as handle:
            # Reject index/file disagreement rather than publishing hidden duplicate keys.
            if set(handle.keys()) != {name for name, filename in source_map.items() if filename == shard}:
                raise ValueError("Verifier shard contains tensors outside its index: " + shard)
            base = {name: handle.get_tensor(name) for name in handle.keys() if not name.startswith("mtp.")}
        if not base:
            continue
        save_file(base, str(output / shard), metadata={"format": "pt"})
        for name, value in base.items():
            result_map[name] = shard
            total_size += value.numel() * value.element_size()
        del base
    save_file(mtp, str(output / mtp_name), metadata={"format": "pt"})
    for name, value in mtp.items():
        result_map[name] = mtp_name
        total_size += value.numel() * value.element_size()
    (output / index_file.name).write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": result_map,
    }, indent=2, sort_keys=True) + "\n")
    result = {"schema_version": "aspec.native-mtp-stitch/v1", "target": str(target),
              "draft": str(draft), "replaced_tensors": sorted(mtp),
              "verifier_tensor_count": len(result_map) - len(mtp),
              "duplicate_mtp_keys": False,
              "files": {name: _sha256(output / name) for name in sorted(set(result_map.values()) | {"config.json"})}}
    (output / "stitch_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stitch_native_mtp(target=args.target, draft=args.draft, output=args.output), indent=2))


if __name__ == "__main__":
    main()
