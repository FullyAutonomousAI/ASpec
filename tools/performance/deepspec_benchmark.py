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

"""Measure native DeepSpec DSpark training with shared, GPU-resident real features.

Run this standalone adapter from an isolated DeepSpec checkout/runtime. Only the
offline cache input and initial-weight names are adapted. The upstream trainer's
compile, sampling, loss, accumulation, clipping and optimizer remain in use.
This is a Draft-only control, not an online Target-to-Draft benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections import deque
from pathlib import Path

KEYMAP = {
    "context_proj.weight": "fc.weight",
    "context_norm.weight": "hidden_norm.weight",
    "final_norm.weight": "norm.weight",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_state(state):
    missing = set(KEYMAP) - state.keys()
    collisions = set(KEYMAP.values()) & state.keys()
    if missing or collisions:
        raise ValueError(f"initial weight keys: missing={sorted(missing)}, collisions={sorted(collisions)}")
    return {KEYMAP.get(key, key): value for key, value in state.items()}


def adapt_feature(raw):
    required = {"input_ids", "loss_mask", "hidden_states", "last_hidden_states"}
    if missing := required - raw.keys():
        raise ValueError(f"feature record lacks {sorted(missing)}")
    converted = {}
    for source, destination in (
        ("input_ids", "input_ids"), ("loss_mask", "loss_mask"),
        ("hidden_states", "target_hidden_states"),
        ("last_hidden_states", "target_last_hidden_states"),
    ):
        tensor = raw[source]
        expected_ndim = 1 if source in {"input_ids", "loss_mask"} else 2
        if tensor.ndim == expected_ndim + 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != expected_ndim:
            raise ValueError(f"invalid {source} shape: {tuple(tensor.shape)}")
        converted[destination] = tensor
    length = converted["input_ids"].shape[0]
    if not length or any(value.shape[0] != length for value in converted.values()):
        raise ValueError("feature token lengths differ or are empty")
    return converted


def resident_batch(feature):
    """Batch-one views equivalent to unpadded CacheCollator, without a CPU roundtrip."""
    return {key: value.unsqueeze(0) for key, value in feature.items()}


def run(args):
    import deepspec
    import deepspec.modeling.dspark.loss as native_loss
    import deepspec.trainer.base_trainer as native_base
    import deepspec.utils.training_logger as native_logger
    import torch
    import torch.distributed as dist
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
    from deepspec.utils import seed_all
    from deepspec.utils.config import (
        CustomJSONEncoder,
        load_config,
        parse_opts_to_config,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3Config

    checkout = args.checkout.resolve()
    if Path(deepspec.__file__).resolve().parent != checkout / "deepspec":
        raise ValueError(f"wrong DeepSpec module origin: {deepspec.__file__}")
    if torch.cuda.device_count() != 1:
        raise ValueError("Draft-only control requires exactly one visible GPU (CP1)")
    if min(args.steps, args.accumulation_steps) < 1 or args.warmup < 1:
        raise ValueError("positive measurement, accumulation and warmup counts are required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.features / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["documents"]
    if not records or len({record["data_id"] for record in records}) != len(records):
        raise ValueError("empty or duplicate feature identities")
    if manifest.get("last_hidden_states_prenorm") is not False:
        raise ValueError("DeepSpec requires normalized final Target hidden states")
    geometry = manifest["draft_config"]
    if (geometry["num_hidden_layers"], geometry["hidden_size"], geometry["markov_rank"]) != (5, 4096, 256):
        raise ValueError("this control requires the five-layer Qwen3-8B DSpark Draft")
    if max(int(record["seq_len"]) for record in records) > args.max_length:
        raise ValueError("feature document exceeds the declared length cap")
    reference = json.loads(args.reference_result.read_text())
    reference_records = {record["data_id"]: record for record in reference["features_manifest"]["documents"]}
    if {record["data_id"]: record for record in records} != reference_records:
        raise ValueError("reference and current feature banks differ")
    ordered_ids = reference["document_order"]
    indices = {record["data_id"]: index for index, record in enumerate(records)}
    if len(ordered_ids) != len(records) or set(ordered_ids) != set(indices):
        raise ValueError("reference document order is not a complete feature-bank permutation")
    if (reference["warmup"], reference["steps"], reference["accumulation_steps"], reference["original_documents_per_microbatch"]) != (args.warmup, args.steps, args.accumulation_steps, 1):
        raise ValueError("reference window/batch differs from requested resident control")
    order = [indices[identity] for identity in ordered_ids]

    # Match train.py's normal precision policy; do not tune CPU/thread/kernel settings.
    torch.set_float32_matmul_precision("high")
    seed_all(42)
    overrides = [
        f"model.target_model_name_or_path={json.dumps(manifest['target'])}",
        "model.num_anchors=512", "model.block_size=7", "model.num_draft_layers=5",
        "train.local_batch_size=1", f"train.global_batch_size={args.accumulation_steps}",
        "train.max_train_steps=1000", "train.sharding_strategy=no_shard",
        f"logging.checkpointing_steps={args.warmup + args.steps + 1}",
        f"data.max_length={args.max_length}",
    ]
    cfg = parse_opts_to_config(overrides, load_config(checkout / "config/dspark/dspark_qwen3_8b.py"))
    # Upstream finalize_cfg recomputes output roots, so set these after finalization.
    cfg.logging.checkpoint_dir = str(args.output / "checkpoints")
    cfg.logging.tensorboard_dir = str(args.output / "tensorboard")
    cfg.data.target_cache_path = "benchmark-resident://shared-real-features"
    pending = deque()
    consumed = []
    physical_tokens = []
    losses = []
    supervision = []
    steps = []
    started = finished = None
    frozen_head_source = {}

    class ResidentDataset:
        def __init__(self, cache_dir):
            self.manifest = {
                "target_layer_ids": geometry["target_layer_ids"],
                "hidden_size": geometry["hidden_size"],
                "target_model_name_or_path": manifest["target"],
            }
            self.records = []
            for record in records:
                path = (args.features / record["path"]).resolve()
                if path.parent != args.features.resolve() or sha256(path) != record["sha256"]:
                    raise ValueError(f"feature identity changed: {path}")
                raw = adapt_feature(torch.load(path, mmap=True, weights_only=True))
                if raw["input_ids"].numel() != int(record["seq_len"]):
                    raise ValueError("feature length disagrees with manifest")
                if raw["target_hidden_states"].shape[1] != 5 * 4096 or raw["target_last_hidden_states"].shape[1] != 4096:
                    raise ValueError("feature hidden dimensions disagree with the Draft")
                self.records.append({key: value.to("cuda") for key, value in raw.items()})

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    class ResidentLoader:
        def __init__(self, trainer, offset, count):
            self.trainer = trainer
            self.offset, self.count = offset, count

        def __len__(self):
            return self.count

        def __iter__(self):
            for occurrence in range(self.offset, self.offset + self.count):
                index = order[occurrence % len(order)]
                batch = resident_batch(self.trainer.train_dataset[index])
                pending.append(index)
                yield batch

    class ResidentTrainer(Qwen3DSparkTrainer):
        def build_models(self):
            config = Qwen3Config(**{**geometry, "num_anchors": 512, "block_size": 7})
            config._attn_implementation = "flex_attention"
            model = Qwen3DSparkModel(config)
            state = adapt_state(load_file(str(args.initial), device="cpu"))
            if "lm_head.weight" not in state:
                # ASpec/TorchSpec keep the frozen verifier head outside the
                # Draft checkpoint; DeepSpec places it inside the model.
                target = Path(manifest["target"]).resolve()
                index = target / "model.safetensors.index.json"
                shard_name = (
                    json.loads(index.read_text())["weight_map"]["lm_head.weight"]
                    if index.is_file() else "model.safetensors"
                )
                shard = (target / shard_name).resolve()
                if shard.parent != target:
                    raise ValueError("frozen Target head shard escapes the model directory")
                with safe_open(shard, framework="pt", device="cpu") as reader:
                    state["lm_head.weight"] = reader.get_tensor("lm_head.weight")
                frozen_head_source.update(
                    path=str(shard), sha256=sha256(shard), key="lm_head.weight",
                    index_sha256=sha256(index) if index.is_file() else None,
                )
            model.load_state_dict(state, strict=True)
            model.set_embedding_head_trainable(False)
            return model.to(device=self.device, dtype=self.precision_dtype), None

        def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
            return ResidentLoader(self, start_offset_samples, num_samples)

        def run_batch(self, batch):
            consumed.append(pending.popleft())
            physical_tokens.append(batch["input_ids"].numel())
            loss = super().run_batch(batch)
            losses.append(loss.detach())
            return loss

        def save_and_eval_checkpoint(self):
            # Final saving/evaluation is outside the resident compute boundary.
            if self.global_step != self.max_train_steps:
                raise RuntimeError("unexpected in-window checkpoint")

    original_dataset = native_base.CacheDataset
    original_log = native_logger.on_optimizer_step
    original_metric = native_loss.add_metric

    def observe_metric(name, value, **kwargs):
        # Reuse already-computed unweighted position counts; no observer reduction/sync.
        if name.startswith("accept_rate@"):
            supervision.append((int(name.split("@")[1]), kwargs["den"].detach()))
        return original_metric(name, value, **kwargs)

    def observe_step(**kwargs):
        nonlocal started, finished
        values = original_log(**kwargs)
        step = int(kwargs["global_step"])
        steps.append({"step": step, "grad_norm": kwargs["grad_norm"], "metrics": values})
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        elif step == args.warmup + args.steps:
            torch.cuda.synchronize()
            finished = time.perf_counter()

    trainer = None
    try:
        native_base.CacheDataset = ResidentDataset
        native_logger.on_optimizer_step = observe_step
        native_loss.add_metric = observe_metric
        trainer = ResidentTrainer(0, cfg)
        native_base.CacheDataset = original_dataset
        # Bound consumption without shortening the shared 1,000-step LR horizon.
        trainer.max_train_steps = args.warmup + args.steps
        trainer.train()
        expected = (args.warmup + args.steps) * args.accumulation_steps
        if len(steps) != args.warmup + args.steps or len(consumed) != expected or pending:
            raise ValueError("incomplete native optimizer or document consumption")
        if len(supervision) != expected * 7 or any(position != i % 7 for i, (position, _) in enumerate(supervision)):
            raise ValueError("missing native per-position supervision counts")
        if started is None or finished is None or not finished > started:
            raise ValueError("missing positive continuous optimizer window")
        loss_values = torch.stack(losses).float().cpu().tolist()
        if len(loss_values) != expected or not all(math.isfinite(value) for value in loss_values):
            raise ValueError("missing/nonfinite native microbatch loss")
        if not all(math.isfinite(float(step["grad_norm"])) for step in steps):
            raise ValueError("nonfinite native optimizer gradient norm")
        counts = torch.stack([value for _, value in supervision]).reshape(expected, 7).cpu()
        measured = consumed[args.warmup * args.accumulation_steps:]
        tokens = sum(int(records[index]["seq_len"]) for index in measured)
        predictions = int(counts[args.warmup * args.accumulation_steps:].sum().item())
        elapsed = finished - started
        result = {
            "schema": "aspec.deepspec-draft-control/v1", "framework": "deepspec",
            "scope": "GPU-resident Draft-only; excludes Target generation, disk and H2D; not online E2E",
            "input_adapter": "batch-one GPU views replace native CPU cache collation; training math unchanged",
            "checkout": str(checkout), "revision": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
            "module_origin": deepspec.__file__, "adapter_sha256": sha256(__file__),
            "features_manifest": manifest, "features_manifest_sha256": sha256(manifest_path),
            "reference_result_sha256": sha256(args.reference_result),
            "document_ordering": "recorded TorchSpec resident-control order; native anchor sampler unchanged",
            "initial_sha256": sha256(args.initial), "initial_keymap": KEYMAP,
            "frozen_target_head": frozen_head_source,
            "native_config": cfg, "draft_config": trainer.draft_model.config.to_dict(),
            "warmup_steps": args.warmup, "measured_steps": args.steps,
            "accumulation_steps": args.accumulation_steps, "anchor_slots_per_physical_sequence": 512,
            "documents": len(measured), "input_tokens": tokens, "supervised_predictions": predictions,
            "physical_tokens": sum(physical_tokens[args.warmup * args.accumulation_steps:]),
            "wall_seconds": elapsed, "input_tokens_per_second": tokens / elapsed,
            "supervised_predictions_per_second": predictions / elapsed,
            "measured_document_order": [records[index]["data_id"] for index in measured],
            "native_step_metrics": steps,
            "native_microbatch_losses": loss_values,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "environment": {"hostname": socket.gethostname(), "platform": platform.platform(),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                            "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2, cls=CustomJSONEncoder) + "\n")
        return result
    finally:
        native_base.CacheDataset = original_dataset
        native_logger.on_optimizer_step = original_log
        native_loss.add_metric = original_metric
        native_logger.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True, help="Common ASpec/TorchSpec model.safetensors")
    parser.add_argument("--reference-result", type=Path, required=True, help="Matched TorchSpec resident-control JSON for feature/order/window identity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, choices=(4096, 131072), required=True)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("wall_seconds", "input_tokens_per_second", "supervised_predictions_per_second")}))


if __name__ == "__main__":
    main()
