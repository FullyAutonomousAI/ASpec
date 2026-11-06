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

"""Observe native DSpark end-to-end runs, without replacing training or scheduling.

The Ray job hook adds counters after native DSpark metric synchronization. No
extra distributed collective, anchor injection, loss change or scheduling hook
is used. Driver time spans real optimizer completions, not summed compute time.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from tools.performance.torchspec_benchmark import package_version, sha256

_INPUT_ENV = (
    "BENCH_TARGET_MODEL",
    "BENCH_INITIAL_DRAFT",
    "BENCH_TRAIN_DATA",
    "BENCH_OUTPUT",
    "BENCH_CACHE",
    "TORCHSPEC_CHECKOUT",
)
_RUNTIME_ENV = (
    "CC",
    "CXX",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_FRAMEWORKS = ("aspec", "torchspec", "angelspec")


def invoke_native_entry(framework, entry):
    """Observe each framework's public entry without replacing its runtime."""
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported benchmark framework: {framework}")
    if framework == "aspec":
        return entry.main()
    return entry.train_async_no_generation(entry.parse_config())


def worker_pythonpath(environment=None):
    """Make the benchmark observer importable inside clean Ray workers."""

    environment = os.environ if environment is None else environment
    entries = [str(_PROJECT_ROOT)]
    entries.extend(
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry not in entries
    )
    return os.pathsep.join(entries)


def _run_identity_command(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _nvidia_smi_identity():
    result = _run_identity_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.get("returncode") == 0:
        return result
    result["fallback_without_compute_cap"] = _run_identity_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return result


def _path_identity(raw_path):
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    identity = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return identity
    if path.is_file():
        identity.update(kind="file", size_bytes=path.stat().st_size, sha256=sha256(path))
        sidecars = []
        for name in ("manifest.json", "source_manifest.json"):
            sidecar = path.parent / name
            if sidecar.is_file():
                sidecars.append(
                    {
                        "path": str(sidecar),
                        "size_bytes": sidecar.stat().st_size,
                        "sha256": sha256(sidecar),
                    }
                )
        identity["sidecars"] = sidecars
        return identity

    candidates = []
    for pattern in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer_config.json",
        "*.safetensors",
        "*.bin",
    ):
        candidates.extend(path.glob(pattern))
    files = []
    for candidate in sorted(set(candidates)):
        if not candidate.is_file():
            continue
        item = {
            "relative_path": str(candidate.relative_to(path)),
            "size_bytes": candidate.stat().st_size,
        }
        # Hash metadata and weights up to 2 GiB. Larger weights retain an
        # inventory, not a content hash; do not present that inventory as
        # cryptographic proof of equal weights.
        if candidate.stat().st_size <= 2 * 1024**3 or candidate.suffix == ".json":
            item["sha256"] = sha256(candidate)
        files.append(item)
    identity.update(kind="directory", files=files)
    return identity


def benchmark_identity(invocation):
    input_environment = {key: os.environ.get(key) for key in _INPUT_ENV}
    return {
        "invocation": invocation,
        "working_directory": str(Path.cwd()),
        "input_environment": input_environment,
        "inputs": {
            key: _path_identity(value)
            for key, value in input_environment.items()
            if key in {"BENCH_TARGET_MODEL", "BENCH_INITIAL_DRAFT", "BENCH_TRAIN_DATA"}
        },
        "runtime_environment": {key: os.environ.get(key) for key in _RUNTIME_ENV},
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "nvidia_smi": _nvidia_smi_identity(),
            "python_packages": _run_identity_command(
                [sys.executable, "-m", "pip", "freeze", "--all"]
            ),
        },
    }


def source_control_identity(module_file):
    module_path = Path(module_file).resolve()
    candidate = module_path.parent
    top_level = _run_identity_command(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"]
    )
    root = Path(top_level["stdout"]) if top_level.get("returncode") == 0 else candidate
    revision = _run_identity_command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run_identity_command(["git", "-C", str(root), "status", "--short"])
    return {
        "root": str(root),
        "revision": revision,
        "dirty": bool(status.get("stdout")) if status.get("returncode") == 0 else None,
        "status": status,
    }


def count_batch_documents(batch):
    """Count source documents, not physical rows produced by THD packing."""

    work = batch.get("_aspec_work")
    if work is not None:
        if not isinstance(work, (list, tuple)) or not work:
            raise ValueError("Native work identity must be a non-empty sequence")
        return len(work)
    return int(batch["input_ids"].shape[0])


def batch_work_ids(batch):
    """Return stable semantic identities when the native runtime exposes them."""

    work = batch.get("_aspec_work")
    if work is None:
        return None
    if not isinstance(work, (list, tuple)) or not work:
        raise ValueError("Native work identity must be a non-empty sequence")
    identities = []
    for item in work:
        work_id = item.get("work_id") if isinstance(item, dict) else None
        if not isinstance(work_id, str) or not work_id:
            raise ValueError("Native work item is missing a stable work_id")
        identities.append(work_id)
    return identities


def observe_padded_collator_inputs(collator_class):
    """Count unpadded source shapes before collation, without a CUDA reduction."""
    original = collator_class.__call__

    def observed(self, features):
        # Cached/USP samples may already contain padding. Their masks, rather
        # than physical shapes, remain authoritative in the fallback observer.
        count = (
            sum(math.prod(item["input_ids"].shape) for item in features)
            if not getattr(self, "usp_enabled", False)
            and all(item.get("attention_mask") is None for item in features)
            else None
        )
        batch = original(self, features)
        if count is not None:
            batch["_benchmark_input_tokens_cpu"] = count
        return batch

    collator_class.__call__ = observed


def install_actor_observers():
    import torch
    namespace = os.environ["ASPEC_BENCH_NAMESPACE"]
    cls = importlib.import_module(namespace + ".training.dspark_trainer").DSparkTrainer
    if getattr(cls, "_benchmark_observed", False):
        return
    cudnn_observation = None
    if namespace == "aspec":
        from tools.performance.torchspec_benchmark import observe_cudnn_attention

        cudnn_observation = observe_cudnn_attention()
        observe_padded_collator_inputs(
            importlib.import_module("aspec.data.utils").DataCollatorWithPadding
        )
    original_micro = cls._train_step
    original_aggregate = cls._aggregate_metrics
    original_batches = cls._iter_batches_from_queue
    # Resolve once in the actual actor, not from the driver's assumed checkout.
    # Hashing is outside the timed microbatch loop and never changes training.
    actor_sources = {}
    for suffix in ("models.dspark", "training.dspark_trainer"):
        path = importlib.import_module(namespace + "." + suffix).__file__
        actor_sources[suffix] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    if namespace == "aspec":
        root = Path(importlib.import_module(namespace).__file__).parent
        for suffix in ("models.ops.loss", "models.ops.tiled_l1"):
            path = root / (suffix.replace(".", "/") + ".py")
            if path.is_file():
                actor_sources[suffix] = {"path": str(path.resolve()), "sha256": sha256(path)}

    def batches(self, *args, **kwargs):
        for batch in original_batches(self, *args, **kwargs):
            # The native training loop removes work identity before calling
            # ``_train_step``.  Preserve only its cardinality so a packed THD
            # row is not mislabeled as one source document.
            batch["_benchmark_documents"] = count_batch_documents(batch)
            batch["_benchmark_work_ids"] = batch_work_ids(batch)
            yield batch

    def micro(self, batch, *args, **kwargs):
        documents = batch.pop("_benchmark_documents", None)
        work_ids = batch.pop("_benchmark_work_ids", None)
        input_tokens_cpu = batch.pop("_benchmark_input_tokens_cpu", None)
        before = dict(cudnn_observation) if cudnn_observation is not None else {}
        result = original_micro(self, batch, *args, **kwargs)
        result["_benchmark_cudnn"] = {
            key: cudnn_observation[key] - before[key]
            for key in ("forward_calls", "backward_calls")
        } if cudnn_observation is not None else {}
        if getattr(self.args, "attention_backend", None) == "cudnn_flex_attention":
            expected = len(self.draft_model.layers)
            if any(value != expected for value in result["_benchmark_cudnn"].values()):
                raise RuntimeError("cuDNN forward/backward was not observed on every Draft layer")
        result["_benchmark_physical_tokens"] = int(batch["input_ids"].numel())
        result["_benchmark_row_length"] = int(batch["input_ids"].shape[-1])
        mask = batch.get("attention_mask")
        result["_benchmark_input"] = (
            input_tokens_cpu if input_tokens_cpu is not None else
            mask.sum().detach() if isinstance(mask, torch.Tensor)
            else int(batch["input_ids"].numel())
        )
        result["_benchmark_query_shapes"] = {
            key: (int(batch["input_ids"].shape[0]) * value
                  if isinstance(value, int) else None)
            for key, value in (
                ("backbone_slots", getattr(self.model, "backbone_blocks_observed", None)),
                ("head_slots", getattr(self.model, "loss_head_blocks_observed", None)),
            )
        }
        result["_benchmark_documents"] = (
            int(documents)
            if documents is not None
            else count_batch_documents(batch)
        )
        result["_benchmark_work_ids"] = work_ids
        result["_benchmark_anchor_slots"] = (
            result["_benchmark_documents"]
            if getattr(self.args, "anchor_sampling_scope", "row") == "sample"
            else int(batch["input_ids"].shape[0])
        ) * int(self.num_anchors)
        result["_benchmark_l1_kernel_invocations"] = dict(
            getattr(self.model, "l1_kernel_invocations", {})
        )
        result["_benchmark_runtime"] = {
            "actor_sources": actor_sources,
            "anchor_sampling_scope": getattr(self.args, "anchor_sampling_scope", "row"),
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "block_size": int(self.block_size),
            "l1_loss_backend": getattr(self.model, "l1_loss_backend", None),
            "fused_clip_scaling": getattr(self.optimizer, "_fuse_clip_scaling", False),
            "qk_prepare_backends": [
                getattr(layer.self_attn, "qk_prepare_backend", None)
                for layer in self.draft_model.layers
            ],
            "cudnn_qkv_format": (
                cudnn_observation["qkv_format"] if cudnn_observation is not None else None
            ),
            "runtime_environment": {key: os.environ.get(key) for key in _RUNTIME_ENV},
        }
        return result

    def aggregate(self, all_step_metrics, step, **kwargs):
        result = original_aggregate(self, all_step_metrics, step, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("This observer requires native immediate DSpark metrics")
        if not all_step_metrics:
            return result
        # Native loss/accuracy materialization has already synchronized these
        # steps. Preserve that location; do not move GPU fences into microsteps.
        predictions = torch.stack([m["count_per_position"] for m in all_step_metrics]).sum().item()
        observed_tokens = [m["_benchmark_input"] for m in all_step_metrics]
        tokens = sum(int(value) for value in observed_tokens if not isinstance(value, torch.Tensor))
        tensor_tokens = [value for value in observed_tokens if isinstance(value, torch.Tensor)]
        if tensor_tokens:
            # Keep the native fallback to one reduction/materialization per
            # optimizer step, rather than one GPU read per microbatch.
            tokens += torch.stack(tensor_tokens).sum().item()
        runtimes = [m["_benchmark_runtime"] for m in all_step_metrics]
        if any(runtime != runtimes[0] for runtime in runtimes[1:]):
            raise RuntimeError("Actor runtime identity changed within one optimizer step")
        result["_benchmark"] = {
            "step": step + 1, "rank": torch.distributed.get_rank(),
            "dp_rank": self.dp_rank,
            "cp_rank": getattr(self, "cp_rank", 0),
            "cp_size": getattr(self, "cp_size", 1),
            "documents": sum(m["_benchmark_documents"] for m in all_step_metrics),
            "work_ids": (
                [
                    work_id
                    for metric in all_step_metrics
                    for work_id in (metric.get("_benchmark_work_ids") or [])
                ]
                if any(
                    metric.get("_benchmark_work_ids") is not None
                    for metric in all_step_metrics
                )
                else None
            ),
            "input_tokens": int(tokens), "supervised_predictions": int(predictions),
            "anchor_slots": sum(m["_benchmark_anchor_slots"] for m in all_step_metrics),
            "physical_tokens": sum(m["_benchmark_physical_tokens"] for m in all_step_metrics),
            "query_shapes": {
                key: (sum(m["_benchmark_query_shapes"][key] for m in all_step_metrics)
                      if all(m["_benchmark_query_shapes"][key] is not None for m in all_step_metrics)
                      else None)
                for key in ("backbone_slots", "head_slots")
            },
            "row_lengths": [m["_benchmark_row_length"] for m in all_step_metrics],
            "row_documents": [m["_benchmark_documents"] for m in all_step_metrics],
            "cudnn_calls": {
                key: sum(m["_benchmark_cudnn"].get(key, 0) for m in all_step_metrics)
                for key in ("forward_calls", "backward_calls")
            },
            "runtime": runtimes[0],
            # A cumulative observation, not immutable actor identity. Snapshot
            # the final microbatch; summing cumulative snapshots double-counts.
            "l1_kernel_invocations": all_step_metrics[-1].get("_benchmark_l1_kernel_invocations", {}),
            "memory_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "memory_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loss": result["train/avg_loss"], "grad_norm": result["train/grad_norm"],
        }
        return result

    cls._train_step = micro
    cls._aggregate_metrics = aggregate
    cls._iter_batches_from_queue = batches
    cls._benchmark_observed = True


def combine_rank_metrics(results, *, expected_ranks=None):
    rows = [
        {**result["_benchmark"], "native_timing": {
            key: value for key, value in result.items()
            if key.startswith("perf/") and isinstance(value, (int, float))
        }}
        for result in results
    ]
    if not rows or len({r["step"] for r in rows}) != 1:
        raise ValueError("Missing or mismatched optimizer step")
    if len({r["rank"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trainer rank")
    if expected_ranks is not None and {r["rank"] for r in rows} != set(expected_ranks):
        raise ValueError("Missing or unexpected trainer ranks")
    if not all(math.isfinite(r[k]) for r in rows for k in ("loss", "grad_norm")):
        raise ValueError("Nonfinite native loss/gradient")
    block_sizes = {r["runtime"]["block_size"] for r in rows}
    if len(block_sizes) != 1:
        raise ValueError("Trainer ranks disagree on the Draft block size")
    cp_sizes = {r.get("cp_size", 1) for r in rows}
    if len(cp_sizes) != 1 or next(iter(cp_sizes)) < 1:
        raise ValueError("Trainer ranks disagree on context parallel size")
    cp_size = cp_sizes.pop()
    groups = {}
    for row in rows:
        groups.setdefault(row.get("dp_rank", row["rank"]), []).append(row)
    leaders = []
    for group in groups.values():
        if len(group) != cp_size or {r.get("cp_rank", 0) for r in group} != set(range(cp_size)):
            raise ValueError("Missing or duplicated context parallel rank")
        leader = next(r for r in group if r.get("cp_rank", 0) == 0)
        # CP ranks carry replicated input IDs/work identity but disjoint query
        # supervision. Count source work once; sum predictions and kernel calls.
        for row in group:
            if any(row.get(key) != leader.get(key) for key in (
                "documents", "input_tokens", "physical_tokens", "anchor_slots", "work_ids",
            )):
                raise ValueError("Context parallel ranks disagree on source work")
        leaders.append(leader)
    return {
        "step": rows[0]["step"],
        **{
            k: sum(r[k] for r in leaders)
            for k in ("documents", "input_tokens", "anchor_slots")
        },
        "supervised_predictions": sum(r["supervised_predictions"] for r in rows),
        "physical_tokens": sum(r.get("physical_tokens", 0) for r in leaders),
        "cudnn_calls": {
            key: sum(r.get("cudnn_calls", {}).get(key, 0) for r in rows)
            for key in ("forward_calls", "backward_calls")
        },
        "block_size": block_sizes.pop(),
        "ranks": rows,
        "work_ids": (
            [work_id for row in leaders for work_id in row["work_ids"]]
            if all(row.get("work_ids") is not None for row in leaders)
            else None
        ),
    }


class NativeObserver:
    def __init__(
        self,
        ray,
        warmup,
        *,
        expected_documents=None,
        expected_input_tokens=None,
        expected_requested_documents=None,
        max_dropped_documents=None,
    ):
        self.ray, self.warmup = ray, warmup
        self.original_get = ray.get
        self.active = False
        self.records = []
        self.boundaries = {}
        self.controller = None
        self.expected_steps = None
        self.expected_ranks = None
        self.expected_documents = expected_documents
        self.expected_input_tokens = expected_input_tokens
        self.expected_requested_documents = expected_requested_documents
        self.max_dropped_documents = max_dropped_documents
        self.training_result = None
        self.work_ids = set()

    def get(self, *args, **kwargs):
        result = self.original_get(*args, **kwargs)
        return self.record(result)

    def record(self, result):
        if self.active and isinstance(result, list) and result and all(
            isinstance(row, dict) and "_benchmark" in row for row in result
        ):
            stamp = time.perf_counter()
            record = combine_rank_metrics(result, expected_ranks=self.expected_ranks)
            if record["step"] in {r["step"] for r in self.records}:
                raise ValueError("Duplicate optimizer completion")
            record["completed_at"] = stamp
            if record["work_ids"] is not None:
                duplicated = self.work_ids.intersection(record["work_ids"])
                if duplicated or len(record["work_ids"]) != len(set(record["work_ids"])):
                    raise ValueError("Duplicate semantic work identity in benchmark run")
                self.work_ids.update(record["work_ids"])
            self.records.append(record)
            if record["step"] == self.warmup:
                self.boundaries["warmup"] = self.original_get(self.controller.get_full_status.remote())
            print(json.dumps({"benchmark_completed_step": record["step"],
                              "documents": record["documents"],
                              "supervised_predictions": record["supervised_predictions"]}), flush=True)
        return result

    def wrap_loop(self, original):
        def observed(args, controller, inference_manager, train_group, *positional, **kwargs):
            self.controller = controller
            self.expected_steps = (
                None
                if getattr(args, "source_work_budget", None) is not None
                else int(args.num_train_steps)
            )
            self.expected_ranks = set(range(len(train_group._actor_handlers)))
            if self.expected_steps is not None and self.expected_steps <= self.warmup:
                raise ValueError("No configured post-warmup window")
            self.started_at = time.perf_counter()
            self.active = True
            try:
                result = original(args, controller, inference_manager, train_group, *positional, **kwargs)
                self.training_result = result
                self.boundaries["end"] = self.original_get(controller.get_full_status.remote())
                self.boundaries["inference_end"] = self.original_get(inference_manager.get_status.remote())
                return result
            finally:
                self.finished_at = time.perf_counter()
                self.active = False
        return observed

    def summarize(self):
        records = sorted(self.records, key=lambda r: r["step"])
        if [r["step"] for r in records] != list(range(1, len(records) + 1)):
            raise ValueError("Optimizer completion gap")
        if self.expected_steps is not None and len(records) != self.expected_steps:
            raise ValueError("Incomplete configured optimizer-step window")
        if len(records) <= self.warmup:
            raise ValueError("No post-warmup window")
        measured = records[self.warmup:]
        wall = measured[-1]["completed_at"] - records[self.warmup - 1]["completed_at"]
        if not all(math.isfinite(record["completed_at"]) for record in records) or wall <= 0 or any(
            later["completed_at"] <= earlier["completed_at"]
            for earlier, later in zip(records, records[1:])
        ):
            raise ValueError("Invalid optimizer-completion clock")
        total = {
            k: sum(r[k] for r in measured)
            for k in ("documents", "input_tokens", "supervised_predictions", "anchor_slots")
        }
        block_sizes = {r["block_size"] for r in measured}
        if len(block_sizes) != 1:
            raise ValueError("Draft block size changed inside the measured window")
        block_size = block_sizes.pop()
        supervision_capacity = total["anchor_slots"] * block_size
        physical_tokens = sum(r["physical_tokens"] for r in measured)
        all_documents = sum(r["documents"] for r in records)
        all_input_tokens = sum(r["input_tokens"] for r in records)
        all_supervised_predictions = sum(r["supervised_predictions"] for r in records)
        all_anchor_slots = sum(r["anchor_slots"] for r in records)
        all_physical_tokens = sum(r["physical_tokens"] for r in records)
        if self.expected_documents is not None and all_documents != self.expected_documents:
            raise ValueError(
                f"Corpus document coverage mismatch: expected {self.expected_documents}, "
                f"observed {all_documents}"
            )
        if self.work_ids and len(self.work_ids) != all_documents:
            raise ValueError("Semantic work identity coverage does not match document count")
        if self.expected_input_tokens is not None and all_input_tokens != self.expected_input_tokens:
            raise ValueError(
                f"Corpus input-token coverage mismatch: expected {self.expected_input_tokens}, "
                f"observed {all_input_tokens}"
            )
        finite_source = None
        if self.expected_requested_documents is not None:
            training = self.training_result
            if not isinstance(training, dict):
                raise ValueError("Finite-source benchmark did not return training evidence")
            requested = training.get("requested_source_work_budget")
            effective = training.get("effective_source_work")
            committed = training.get("committed_source_work")
            dropped = training.get("dropped_source_work")
            policy = training.get("source_work_tail_policy")
            if requested != self.expected_requested_documents:
                raise ValueError(
                    "Finite-source request mismatch: "
                    f"expected {self.expected_requested_documents}, observed {requested}"
                )
            if effective != all_documents or committed != all_documents:
                raise ValueError(
                    "Finite-source effective work does not match observed documents: "
                    f"effective={effective}, committed={committed}, observed={all_documents}"
                )
            if dropped != requested - all_documents:
                raise ValueError(
                    "Finite-source dropped-work accounting mismatch: "
                    f"requested={requested}, observed={all_documents}, dropped={dropped}"
                )
            if (
                self.max_dropped_documents is not None
                and (dropped < 0 or dropped > self.max_dropped_documents)
            ):
                raise ValueError(
                    "Finite-source drop-last bound exceeded: "
                    f"dropped={dropped}, maximum={self.max_dropped_documents}"
                )
            if not isinstance(policy, str) or not policy.startswith("drop_last_"):
                raise ValueError(f"Finite-source tail policy is not drop-last: {policy!r}")
            finite_source = {
                "requested_documents": requested,
                "effective_documents": all_documents,
                "dropped_documents": dropped,
                "max_dropped_documents": self.max_dropped_documents,
                "tail_policy": policy,
            }
        full_wall = records[-1]["completed_at"] - self.started_at
        if full_wall <= 0:
            raise ValueError("Invalid full-corpus optimizer-completion clock")
        summary = {
            "warmup_steps": self.warmup, "measured_steps": len(measured),
            "wall_seconds": wall, **total,
            **{k + "_per_second": v / wall for k, v in total.items()},
            "draft_block_size": block_size,
            "supervision_capacity_positions": supervision_capacity,
            "supervision_fill_ratio": total["supervised_predictions"] / supervision_capacity,
            "physical_tokens": physical_tokens,
            "padding_efficiency": total["input_tokens"] / physical_tokens if physical_tokens else None,
            "cudnn_calls": {
                key: sum(r["cudnn_calls"][key] for r in measured)
                for key in ("forward_calls", "backward_calls")
            },
            "whole_loop_seconds": self.finished_at - self.started_at,
            "full_corpus": {
                "documents": all_documents,
                "input_tokens": all_input_tokens,
                "supervised_predictions": all_supervised_predictions,
                "anchor_slots": all_anchor_slots,
                "physical_tokens": all_physical_tokens,
                "optimizer_steps": len(records),
                "time_to_last_optimizer_completion_seconds": full_wall,
                "documents_per_second": all_documents / full_wall,
                "input_tokens_per_second": all_input_tokens / full_wall,
                "supervised_predictions_per_second": all_supervised_predictions / full_wall,
                "padding_efficiency": (
                    all_input_tokens / all_physical_tokens if all_physical_tokens else None
                ),
                "unique_work_ids": len(self.work_ids) if self.work_ids else None,
                "work_ids_sha256": (
                    hashlib.sha256("\n".join(sorted(self.work_ids)).encode()).hexdigest()
                    if self.work_ids
                    else None
                ),
            },
            "window": "completion(warmup) to completion(last), native continuous pipeline",
        }
        if finite_source is not None:
            summary["finite_source"] = finite_source
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=SUPPORTED_FRAMEWORKS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--expected-input-tokens", type=int)
    parser.add_argument("--expected-requested-documents", type=int)
    parser.add_argument("--max-dropped-documents", type=int)
    invocation = [sys.executable, *sys.argv]
    args, overrides = parser.parse_known_args()
    if args.result.exists() or args.warmup < 1:
        parser.error("Use a new result path and positive warmup")
    if args.expected_documents is not None and args.expected_requested_documents is not None:
        parser.error("Choose exact document coverage or finite-source drop-last validation")
    if args.expected_requested_documents is not None and args.expected_input_tokens is not None:
        parser.error("Finite-source packing has data-dependent input-token coverage")
    if args.max_dropped_documents is not None and args.expected_requested_documents is None:
        parser.error("--max-dropped-documents requires --expected-requested-documents")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ASPEC_BENCH_NAMESPACE"] = args.framework
    env_module = importlib.import_module(args.framework + ".utils.env")
    env = getattr(env_module, "get_" + args.framework + "_env_vars")()
    env.update({k: os.environ[k] for k in ("LD_LIBRARY_PATH", "PYTHONPATH", "ASPEC_BENCH_NAMESPACE")
                if k in os.environ})
    env["PYTHONPATH"] = worker_pythonpath()
    import ray

    ray.init(address="local", runtime_env={
        "worker_process_setup_hook": install_actor_observers, "env_vars": env,
    })
    observer = NativeObserver(
        ray,
        args.warmup,
        expected_documents=args.expected_documents,
        expected_input_tokens=args.expected_input_tokens,
        expected_requested_documents=args.expected_requested_documents,
        max_dropped_documents=args.max_dropped_documents,
    )
    loop = importlib.import_module(args.framework + ".controller.loop")
    loop.training_loop = observer.wrap_loop(loop.training_loop)
    if hasattr(loop, "_wait_for_trainer_group"):
        native_wait = loop._wait_for_trainer_group

        def observed_wait(*positional, **kwargs):
            return observer.record(native_wait(*positional, **kwargs))

        loop._wait_for_trainer_group = observed_wait
    ray.get = observer.get
    payload = {"schema": "aspec.torchspec-e2e-benchmark/v1", "framework": args.framework,
               "config": str(args.config.resolve()), "overrides": overrides,
               "host": socket.gethostname(), "python": sys.executable,
               "versions": {k: package_version(k) for k in
                            ("torch", "sglang", "transformers", "flash-attn-4", "ray", "mooncake-transfer-engine")}}
    payload["benchmark_identity"] = benchmark_identity(invocation)
    payload["ray_cluster_resources"] = ray.cluster_resources()
    payload["ray_worker_environment"] = {
        key: env.get(key) for key in _RUNTIME_ENV if key in env
    }
    from omegaconf import OmegaConf

    resolved_config = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_cli(overrides)
    )
    payload["resolved_config"] = OmegaConf.to_container(resolved_config, resolve=True)
    payload["tool_sha256"] = sha256(__file__)
    payload["config_sha256"] = sha256(args.config)
    payload["framework_source"] = importlib.import_module(args.framework).__file__
    payload["source_control"] = source_control_identity(payload["framework_source"])
    sglang_source = importlib.import_module("sglang").__file__
    payload["sglang_source"] = {
        "path": str(Path(sglang_source).resolve()),
        "sha256": sha256(sglang_source),
        "source_control": source_control_identity(sglang_source),
    }
    payload["source_sha256"] = {
        suffix: sha256(importlib.import_module(args.framework + "." + suffix).__file__)
        for suffix in ("training.dspark_trainer", "training.dflash_trainer", "training.trainer",
                       "controller.training_controller", "controller.loop", "data.utils")
    }
    if args.framework == "aspec":
        for suffix in ("models.dspark", "models.dflash", "models.ops.output_blocks", "models.ops.direct_cp",
                       "models.ops.loss", "training.optimizer"):
            payload["source_sha256"][suffix] = sha256(
                importlib.import_module("aspec." + suffix).__file__
            )
        for suffix in ("models.ops.tiled_l1",):
            kernel_source = Path(payload["framework_source"]).parent / (suffix.replace(".", "/") + ".py")
            if kernel_source.is_file():
                payload["source_sha256"][suffix] = sha256(kernel_source)
    entry = importlib.import_module(args.framework + ".train_entry")
    sys.argv = ["benchmark-native-entry", "--config", str(args.config), *overrides]
    try:
        invoke_native_entry(args.framework, entry)
        payload["summary"] = observer.summarize()
        payload["success"] = True
    except BaseException as error:
        payload["success"] = False
        payload["error"] = repr(error)
        raise
    finally:
        payload["records"] = observer.records
        payload["boundaries"] = observer.boundaries
        args.result.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        ray.get = observer.original_get
        if ray.is_initialized():
            ray.shutdown()
    print(json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
