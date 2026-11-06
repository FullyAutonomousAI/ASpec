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

"""Run a resident-feature Draft-only control through native SpecForge training.

This intentionally mirrors ``torchspec_benchmark.py``'s Draft-only boundary:
real frozen Target features are loaded before timing and remain on the Draft
GPU.  Model construction, DSpark loss, backward, optimizer and scheduling are
the upstream SpecForge implementations; only the feature store is replaced by
a benchmark-local resident store so storage and H2D time are not mislabeled as
Draft compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from tools.performance.specforge_e2e_benchmark import (
    _source_identity,
    _system_identity,
    sha256,
)


def _adapt_feature(raw: dict) -> dict:
    """Translate the shared benchmark feature names to SpecForge's contract."""

    required = {"input_ids", "loss_mask", "hidden_states", "last_hidden_states"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"shared feature record lacks {sorted(missing)}")
    input_ids = raw["input_ids"]
    loss_mask = raw["loss_mask"]
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze(0)
    if loss_mask.ndim == 2 and loss_mask.shape[0] == 1:
        loss_mask = loss_mask.squeeze(0)
    if input_ids.ndim != 1 or loss_mask.ndim != 1:
        raise ValueError("input_ids and loss_mask must normalize to one sequence")
    if input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids and loss_mask lengths differ")
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": raw["hidden_states"],
        "target_last_hidden_states": raw["last_hidden_states"],
    }


def _finite_float(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def run(args) -> dict:
    import torch

    if args.result.exists():
        raise FileExistsError(args.result)
    manifest = json.loads((args.features / "manifest.json").read_text())
    records = manifest["documents"]
    if not records:
        raise ValueError("feature manifest is empty")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps positive")
    total_steps = args.warmup + args.steps
    total_samples = total_steps * args.accumulation_steps

    from specforge import launch
    from specforge.application import resolve_run
    from specforge.cli import _bootstrap_single_process_env
    from specforge.config import load_config
    from specforge.distributed import destroy_distributed, init_distributed
    from specforge.runtime.contracts import FeatureHandle, SampleRef
    from specforge.runtime.control_plane import DataFlowController
    from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore
    from specforge.runtime.data_plane.feature_store import FeatureStore, spec_from_tensor
    from specforge.training import assembly

    # Upstream model-config paths are checkout-relative, as they are for the
    # native ``specforge train`` command.
    os.chdir(args.checkout)
    overrides = list(args.overrides) + [
        f"training.max_steps={total_steps}",
        f"training.accumulation_steps={args.accumulation_steps}",
        f"data.max_length={args.max_length}",
        f"output_dir={args.output_dir}",
        f"run_id={args.run_id}",
    ]
    cfg = load_config(str(args.config), overrides)
    resolved = resolve_run(cfg)
    if resolved.algorithm.name != "dspark":
        raise ValueError("resident Draft-only benchmark currently requires DSpark")

    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    _bootstrap_single_process_env()
    init_distributed(
        timeout=cfg.training.dist_timeout,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    failed = True
    try:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        resident = []
        identities = []
        for record in records:
            path = args.features / record["path"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"feature identity changed: {path}")
            raw = torch.load(path, mmap=True, weights_only=True)
            tensors = {
                name: value.to(device=device, non_blocking=False)
                for name, value in _adapt_feature(raw).items()
            }
            length = int(tensors["input_ids"].numel())
            if length != int(record["seq_len"]):
                raise ValueError(f"feature length changed for {record['data_id']}")
            resident.append(tensors)
            identities.append(
                {
                    "data_id": record["data_id"],
                    "input_tokens": length,
                    "supervised_positions": int(tensors["loss_mask"].sum().item()),
                }
            )

        class ResidentStore(FeatureStore):
            def __init__(self):
                self.fetches: list[int] = []

            def consumer_device(self):
                return device

            def put(self, tensors, *, sample_id, metadata):  # pragma: no cover
                raise NotImplementedError

            def get(self, sample_ref, *, device=device, names=None):
                index = int(sample_ref.metadata["resident_index"])
                self.fetches.append(index)
                tensors = resident[index]
                if names is not None:
                    tensors = {name: tensors[name] for name in names}
                return tensors, FeatureHandle(sample_ref.sample_id, 1, "resident")

            def release(self, handle, *, reason="consumed"):
                return None

            def abort(self, sample_id, *, reason):
                return None

            def health(self):
                return {"resident_records": len(resident), "fetches": len(self.fetches)}

        store = ResidentStore()
        refs = []
        for occurrence in range(total_samples):
            index = occurrence % len(resident)
            tensors = resident[index]
            refs.append(
                SampleRef(
                    sample_id=f"resident-{occurrence:06d}",
                    run_id=args.run_id,
                    source_task_id=None,
                    feature_store_uri=f"resident://{args.run_id}/{index}",
                    feature_keys={name: name for name in tensors},
                    feature_specs={
                        name: spec_from_tensor(name, value) for name, value in tensors.items()
                    },
                    strategy="dspark",
                    target_model_version=manifest["target"],
                    num_tokens=identities[index]["input_tokens"],
                    estimated_bytes=sum(
                        value.numel() * value.element_size() for value in tensors.values()
                    ),
                    metadata={"target_repr": "hidden_state", "resident_index": index},
                )
            )

        metrics: list[dict] = []
        completion_times: list[float] = []

        def collect(values, step):
            row = {key: value for key, value in values.items()}
            row["step"] = int(step)
            metrics.append(row)
            if len(metrics) in (args.warmup, total_steps):
                torch.cuda.synchronize(device)
            completion_times.append(time.perf_counter())

        bundle = assembly.build_model_bundle(cfg, algorithm=resolved.algorithm)
        collate_fn, transform = launch._offline_io(
            resolved.algorithm,
            cfg.model.input_modality,
            cfg.data.max_length,
            ttt_length=cfg.training.ttt_length,
            use_usp_preprocess=False,
        )
        controller = DataFlowController(
            args.run_id,
            metadata_store=NoOpMetadataStore(),
            enable_sample_queue=False,
        )
        common = assembly._common_launch_kwargs(cfg, bundle, resolved.algorithm, logger=collect)
        common.pop("modality")
        common.update(
            max_steps=total_steps,
            log_interval=1,
            dataloader_num_workers=0,
        )
        trainer = launch._assemble_trainer(
            controller=controller,
            store=store,
            ref_source={"refs": refs},
            model=bundle.model,
            target_head=(
                bundle.target_head
                if resolved.algorithm.providers.step.uses_external_target_head
                else None
            ),
            collate_fn=collate_fn,
            per_sample_transform=transform,
            durable_ack=False,
            num_epochs=1,
            dataset_size=len(refs),
            **common,
        )
        # The benchmark store deliberately returns immutable resident tensors.
        # Avoid a second device-to-device clone, matching the existing resident
        # ASpec/TorchSpec control rather than timing feature movement.
        trainer._loader.clone_on_fetch = False
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        fit_started = time.perf_counter()
        completed = trainer.fit()
        torch.cuda.synchronize(device)
        if completed != total_steps or len(metrics) != total_steps:
            raise ValueError(
                f"incomplete native run: completed={completed}, metrics={len(metrics)}"
            )
        if len(store.fetches) != total_samples:
            raise ValueError(f"resident fetch count {len(store.fetches)} != {total_samples}")

        measured_metrics = metrics[args.warmup :]
        measured_indices = store.fetches[args.warmup * args.accumulation_steps :]
        native_step_seconds = sum(
            _finite_float(row["perf/optimizer_step_time_s"], "optimizer step time")
            for row in measured_metrics
        )
        measured_started = completion_times[args.warmup - 1] if args.warmup else fit_started
        seconds = completion_times[-1] - measured_started
        if seconds <= 0:
            raise ValueError("resident continuous wall window must be positive")
        input_tokens = sum(identities[index]["input_tokens"] for index in measured_indices)
        supervised = sum(identities[index]["supervised_positions"] for index in measured_indices)
        result = {
            "schema": "aspec.specforge-draft-benchmark/v1",
            "framework": "specforge",
            "boundary": (
                "native SpecForge DSpark forward/backward/optimizer with real frozen "
                "Target features resident on the Draft GPU before timing"
            ),
            "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
            "overrides": overrides,
            "source_control": _source_identity(args.checkout),
            "system": _system_identity(),
            "features": {
                "path": str(args.features.resolve()),
                "manifest_sha256": sha256(args.features / "manifest.json"),
                "resident_records": len(resident),
            },
            "metrics": {
                "warmup_steps": args.warmup,
                "measured_steps": args.steps,
                "accumulation_steps": args.accumulation_steps,
                "documents": len(measured_indices),
                "input_tokens": input_tokens,
                "supervised_positions": supervised,
                "seconds": seconds,
                "native_step_seconds": native_step_seconds,
                "timing_boundary": "continuous optimizer-completion wall clock",
                "documents_per_second": len(measured_indices) / seconds,
                "input_tokens_per_second": input_tokens / seconds,
                "supervised_positions_per_second": supervised / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "native_step_metrics": metrics,
            "measured_document_order": [identities[index]["data_id"] for index in measured_indices],
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        failed = False
        return result
    finally:
        destroy_distributed(abort=failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args)["metrics"], indent=2))


if __name__ == "__main__":
    main()
