# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Async Training Controller for decoupled inference and training.

Data flow:
  load_dataset(args) → Stored Dataset → Prompt Buffer → Inference Manager
    → Sample Pool → Train Queues → Training Workers
  (Stored dataset is retained for epoch reloads and vocab mapping computation.)

Controller manages the tokenized dataset (for epoch reloads and vocab mapping),
prompt metadata, and mooncake keys. Actual inference tensor data is stored in
mooncake; the controller only tracks keys and byte sizes for backpressure.

Batch Size Design:
  micro_batch_size                   # Samples per GPU per dispatch (user config)
  per_dp_rank_batch_size             # = micro_batch_size * sp_size (derived)
  dispatch_batch_size            # = per_dp_rank_batch_size * dp_size (samples per dispatch)
  global_batch_size                  # = dispatch_batch_size * accumulation_steps (per optimizer step)

  Example with micro_batch_size=2, sp_size=1, dp_size=4, accumulation_steps=2:
    - per_dp_rank_batch_size = 2 * 1 = 2
    - dispatch_batch_size = 2 * 4 = 8
    - global_batch_size = 8 * 2 = 16

  Data flow per optimizer step (with accumulation_steps=2):
    1. Controller dispatches 8 samples per dispatch, 2 dispatches per optimizer step
    2. Each DP rank receives 2 samples per dispatch (4 total per optimizer step)
    3. Each train actor calls train_from_queue(num_batches=accumulation_steps)
    4. Forward/backward for each micro-batch, optimizer step after last one
"""

import copy
import hashlib
import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import islice

import ray
from ray.util.queue import Queue

from aspec.config.draft_execution import is_direct_cp_backend
from aspec.controller.packing import (
    PackingResult,
    count_eagle_supervised_positions,
    count_eligible_anchor_starts,
    pack_work_aware_rows,
)
from aspec.data.utils import length_grouped_order
from aspec.training.data_fetcher import TrainSample
from aspec.utils.logging import logger
from aspec.utils.memory import estimate_tensor_bytes
from aspec.utils.types import InferenceInput, InferenceOutput
from aspec.work_lifecycle import (
    EXECUTION_ID_KEY,
    WORK_ID_KEY,
    WORK_KIND_KEY,
    WorkLifecycleLedger,
    build_work_identity,
    require_work_identity,
)

_estimate_bytes = estimate_tensor_bytes


def _tensor_fingerprint(value) -> dict | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").contiguous().numpy()
    if hasattr(value, "tobytes"):
        payload = value.tobytes()
        return {
            "dtype": str(getattr(value, "dtype", "unknown")),
            "shape": list(getattr(value, "shape", ())),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"sha256": hashlib.sha256(bytes(value)).hexdigest()}
    return {"value": value}


def dataset_fingerprint(dataset: list) -> str:
    """Hash stable source identity and training-relevant row content."""

    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise ValueError(
                "Exact checkpoint resume requires dictionary dataset rows with stable data_id"
            )
        data_id = str(sample.get("data_id") or sample.get("id") or "")
        if not data_id:
            raise ValueError(
                f"Exact checkpoint resume requires a stable data_id for dataset row {index}"
            )
        if data_id in seen_ids:
            raise ValueError(f"Duplicate dataset data_id prevents stable work identity: {data_id}")
        seen_ids.add(data_id)
        record = {
            "data_id": data_id,
            "formatted_prompt": sample.get("formatted_prompt"),
            "input_ids": _tensor_fingerprint(sample.get("input_ids")),
            "metadata": sample.get("metadata") or {},
            "multimodal_inputs": sample.get("multimodal_inputs"),
            "packed_loss_mask": _tensor_fingerprint(sample.get("packed_loss_mask")),
            "prompt": sample.get("prompt"),
        }
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def take_source_balanced_batch(
    samples: deque[InferenceOutput], batch_size: int
) -> list[InferenceOutput] | None:
    """Remove a FIFO-like batch while preferring distinct source engines.

    TransferEngine source rings are owned by inference engines. Releasing a
    training round from as many distinct sources as are currently ready keeps
    one fast producer from filling its ring while completed outputs from other
    sources remain queued.  Source diversity is a scheduling preference, not a
    progress precondition: a bounded pool or a short finite generation budget
    may legitimately contain fewer sources than the training batch width.
    """
    if batch_size < 1:
        raise ValueError("source-balanced batch size must be positive")

    selected_indexes: list[int] = []
    seen_engines: set[str] = set()
    for index, sample in enumerate(samples):
        engine = str((sample.metadata or {}).get("engine_rank", ""))
        if not engine or engine in seen_engines:
            continue
        selected_indexes.append(index)
        seen_engines.add(engine)
        if len(selected_indexes) == batch_size:
            break

    if len(samples) < batch_size:
        return None

    if len(selected_indexes) < batch_size:
        selected_set = set(selected_indexes)
        for index in range(len(samples)):
            if index in selected_set:
                continue
            selected_indexes.append(index)
            selected_set.add(index)
            if len(selected_indexes) == batch_size:
                break

    selected_set = set(selected_indexes)
    selected = [sample for index, sample in enumerate(samples) if index in selected_set]
    remaining = [sample for index, sample in enumerate(samples) if index not in selected_set]
    samples.clear()
    samples.extend(remaining)
    return selected


def take_source_aware_batch(
    samples: deque[InferenceOutput], batch_size: int
) -> list[InferenceOutput] | None:
    """Keep a complete producer request together, otherwise balance sources.

    A target request can return multiple samples at once.  Splitting those
    adjacent results across training micro-batches adds head-of-line latency
    and delays release of the producer's source-ring slots.  Consume an intact
    request atomically when it exactly fills a training dispatch, otherwise
    prefer distinct sources.  If the available tail has only one source, FIFO
    fallback is required for progress and releases that source's ring slots.
    """
    if batch_size < 1:
        raise ValueError("source-aware batch size must be positive")
    if len(samples) >= batch_size:
        candidates = list(islice(samples, 0, batch_size))
        metadata = [sample.metadata or {} for sample in candidates]
        request_ids = [str(item.get("request_id", "")) for item in metadata]
        request_groups = [request_id.rpartition("_") for request_id in request_ids]
        engine_ranks = [
            "" if item.get("engine_rank") is None else str(item["engine_rank"])
            for item in metadata
        ]
        next_group: tuple[str, str, str] | None = None
        next_rank = ""
        if len(samples) > batch_size:
            next_metadata = samples[batch_size].metadata or {}
            next_group = str(next_metadata.get("request_id", "")).rpartition("_")
            next_rank = (
                ""
                if next_metadata.get("engine_rank") is None
                else str(next_metadata["engine_rank"])
            )
        if (
            request_groups[0][0]
            and request_groups[0][1] == "_"
            and sorted(int(group[2]) for group in request_groups if group[2].isdigit())
            == list(range(batch_size))
            and all(
                group[0] == request_groups[0][0]
                and group[1] == "_"
                and group[2].isdigit()
                for group in request_groups
            )
            and engine_ranks[0]
            and all(rank == engine_ranks[0] for rank in engine_ranks)
            and not (
                next_group is not None
                and next_group[0] == request_groups[0][0]
                and next_group[1] == "_"
                and next_group[2].isdigit()
                and next_rank == engine_ranks[0]
            )
        ):
            return [samples.popleft() for _ in range(batch_size)]
    balanced = take_source_balanced_batch(samples, batch_size)
    if balanced is not None or len(samples) < batch_size:
        return balanced
    return [samples.popleft() for _ in range(batch_size)]


def take_node_local_balanced_batch(
    samples: deque[InferenceOutput], required_by_node: dict[str, int]
) -> list[InferenceOutput] | None:
    """Remove one complete locality-safe DP round from a ready pool.

    A node must contribute enough outputs for its local Draft ranks.  Source
    diversity is still preferred inside each node, but it never permits a
    remote feature fallback.
    """

    if not required_by_node or any(count < 1 for count in required_by_node.values()):
        raise ValueError("node-local dispatch requirements must be positive")
    if len(samples) < sum(required_by_node.values()):
        return None

    indexes_by_node: dict[str, list[int]] = {node: [] for node in required_by_node}
    for index, sample in enumerate(samples):
        metadata = sample.metadata or {}
        if metadata.get("feature_affinity") != "node_local_required":
            raise ValueError(
                f"node-local feature output {sample.data_id!r} does not carry the "
                "required affinity contract"
            )
        node = str(metadata.get("source_node_ip", ""))
        if not node:
            raise ValueError(
                f"node-local feature output {sample.data_id!r} has no source_node_ip"
            )
        if node not in indexes_by_node:
            raise ValueError(
                f"feature output {sample.data_id!r} came from unplanned node {node!r}"
            )
        indexes_by_node[node].append(index)

    if any(
        len(indexes_by_node[node]) < required
        for node, required in required_by_node.items()
    ):
        return None

    selected_indexes: list[int] = []
    for node, required in required_by_node.items():
        node_indexes = indexes_by_node[node]
        node_samples = deque(samples[index] for index in node_indexes)
        chosen_samples = take_source_aware_batch(node_samples, required)
        if chosen_samples is None:
            return None
        chosen_ids = {id(sample) for sample in chosen_samples}
        selected_indexes.extend(
            index for index in node_indexes if id(samples[index]) in chosen_ids
        )

    selected_set = set(selected_indexes)
    selected = [sample for index, sample in enumerate(samples) if index in selected_set]
    remaining = [sample for index, sample in enumerate(samples) if index not in selected_set]
    samples.clear()
    samples.extend(remaining)
    return selected


@dataclass
class SpeedMonitor:
    """Tracks throughput over a sliding time window."""

    window_seconds: float = 10.0
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_count: int = 0

    def record(self, count: int = 1) -> None:
        """Record count entries at current time."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._total_count += count
            self._prune_old_events(now)

    def _prune_old_events(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_speed(self) -> float:
        """Get current speed in entries/sec over the window."""
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            if not self._events:
                return 0.0

            window_count = sum(count for _, count in self._events)
            oldest_time = self._events[0][0]
            elapsed = now - oldest_time
            if elapsed < 0.001:
                return 0.0
            return window_count / elapsed

    def get_total_count(self) -> int:
        """Get total count since start."""
        return self._total_count


class AsyncTrainingController:
    """Central controller for async training pipeline.

    Responsibilities:
      - Loads and stores tokenized datasets for training and eval
      - Computes vocab mappings for draft model pruning
      - Manages prompt buffer (samples waiting for inference)
      - Manages sample pool (completed inferences waiting for training)
      - Dispatches batches to per-DP training queues when pool is full
      - Monitors inference and training throughput
    """

    def __init__(self, args, dp_size: int, *, with_train_queues=True, with_eval_queues=True):
        self.args = args
        self.dp_size = dp_size
        attention_backend = getattr(args, "attention_backend", None)
        if attention_backend == "usp":
            self.sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(
                args, "sp_ring_size", 1
            )
        elif is_direct_cp_backend(attention_backend):
            self.sp_size = getattr(args, "draft_context_parallel_size", 1)
        else:
            self.sp_size = 1
        self.queue_count = dp_size * self.sp_size

        self.prompt_buffer: deque[InferenceInput] = deque()
        self._prompt_lock = threading.Lock()

        self.sample_pool: deque[InferenceOutput] = deque()
        self._pool_lock = threading.Lock()
        self._pool_bytes = 0
        self._sample_bytes: dict[str, int] = {}
        self._lifecycle = WorkLifecycleLedger()
        self._submitted_execution_ids: set[str] = set()

        self.train_queues = [Queue() for _ in range(self.queue_count)] if with_train_queues else []

        # Eval: separate pool and queues so eval data never mixes with training
        self.eval_pool: deque[InferenceOutput] = deque()
        self._eval_pool_lock = threading.Lock()
        self._eval_data_ids: set[str] = set()
        self._eval_expected_count: int = 0
        self._eval_dispatched_samples: int = 0
        self.eval_queues = [Queue() for _ in range(self.queue_count)] if with_eval_queues else []

        self.batch_id = 0
        self.dispatch_batch_size = args.per_dp_rank_batch_size * dp_size
        # Forward-local work is released as before. Only small scheduling loads
        # span dispatches, and reset at an optimizer boundary (including resume).
        self._partition_step: int | None = None
        self._partition_rank_loads = [0] * dp_size
        self._batching_input_tokens = 0
        self._batching_context_tokens = 0
        self._batching_microbatches = 0
        self._source_balanced_dispatch = (
            getattr(args, "mooncake_data_plane", "store") == "transfer_engine"
        )
        self.feature_affinity = str(
            getattr(args, "mooncake_feature_affinity", "any")
        ).lower()
        self._dp_ranks_by_node: dict[str, list[int]] = {}
        self._dp_node_by_rank: dict[int, str] = {}
        if self.feature_affinity == "node_local_required":
            training_nodes = list(getattr(args, "training_node_ips", None) or ())
            gpus_per_node = int(getattr(args, "training_num_gpus_per_node", 0))
            if not training_nodes or gpus_per_node < 1 or gpus_per_node % self.sp_size:
                raise ValueError(
                    "node_local_required needs explicit training nodes and node-local "
                    "Draft CP groups"
                )
            dp_ranks_per_node = gpus_per_node // self.sp_size
            if dp_ranks_per_node * len(training_nodes) != self.dp_size:
                raise ValueError(
                    "node_local_required Draft rank layout does not match dp_size"
                )
            next_rank = 0
            for node in training_nodes:
                ranks = list(range(next_rank, next_rank + dp_ranks_per_node))
                self._dp_ranks_by_node[str(node)] = ranks
                for rank in ranks:
                    self._dp_node_by_rank[rank] = str(node)
                next_rank += dp_ranks_per_node
        elif self.feature_affinity != "any":
            raise ValueError(f"unsupported feature affinity {self.feature_affinity!r}")
        self._node_local_dispatch_samples = 0
        self._cross_node_dispatch_samples = 0
        self._node_local_dispatch_by_node: dict[str, int] = {
            node: 0 for node in self._dp_ranks_by_node
        }
        self.packing_enabled = bool(getattr(args, "dflash_packing", False))
        if self.packing_enabled and getattr(args, "attention_backend", None) == "usp":
            raise NotImplementedError(
                "dflash_packing is not supported with USP"
            )
        self.packing_max_seq = int(getattr(args, "max_seq_length", 0) or 0)
        if self.packing_enabled and self.packing_max_seq <= 0:
            raise ValueError("dflash_packing requires max_seq_length > 0")
        configured_min = int(getattr(args, "packing_min_samples_per_dispatch", 0) or 0)
        self.packing_min_samples_per_dispatch = max(configured_min, dp_size)
        self._row_id_counter = 0
        self._packing_rows_dispatched = 0
        self._packing_documents_dispatched = 0
        self._packing_tokens_dispatched = 0
        self._packing_capacity_tokens = 0
        self._packing_useful_anchor_starts = 0
        self._packing_min_fill_ratio: float | None = None
        self._packing_max_documents_per_row = 0
        self._packing_observed_row_count = 0
        self._packing_observed_row_min: int | None = None
        self._packing_observed_row_max = 0
        self._observed_sequence_count = 0
        self._observed_sequence_min: int | None = None
        self._observed_sequence_max = 0
        self.eval_dispatch_batch_size = dp_size
        self._data_id_counter = 0

        self._stored_dataset: list | None = None
        self._stored_eval_dataset: list | None = None
        self._dataset_fingerprint: str | None = None
        self._dataset_admission: dict = {}
        self._eval_dataset_admission: dict = {}
        self._restored_committed_work_ids: set[str] = set()
        self._dataset_epoch: int = 0
        dataset_seed = getattr(args, "dataset_seed", None)
        self._dataset_seed: int = getattr(args, "seed", 42) if dataset_seed is None else dataset_seed
        self._shuffle_dataset: bool = getattr(args, "shuffle_dataset", True)
        self._length_group_size: int = getattr(args, "length_group_size", 32)

        self._start_time = time.time()
        self._inference_monitor = SpeedMonitor(window_seconds=10.0)
        self._training_monitor = SpeedMonitor(window_seconds=10.0)
        self._last_dispatch_log_time = 0.0
        self._inference_error: str | None = None
        self._shutdown_signaled = False

    def _generate_data_id(self) -> str:
        self._data_id_counter += 1
        return f"data_{self._data_id_counter}"

    def _work_metadata(
        self,
        *,
        data_id: str,
        metadata: dict | None,
        split: str,
        epoch: int,
    ) -> dict:
        result = dict(metadata or {})
        identity = build_work_identity(split=split, epoch=epoch, source_id=data_id)
        for key, value in identity.items():
            configured = result.get(key)
            if configured is not None and configured != value:
                raise ValueError(
                    f"Caller-provided {key} does not match the deterministic ASpec identity"
                )
            result[key] = value
        execution_id = result[EXECUTION_ID_KEY]
        if execution_id in self._submitted_execution_ids:
            raise ValueError(f"Duplicate submitted work identity: {execution_id}")
        self._submitted_execution_ids.add(execution_id)
        return result

    # ─────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────

    def add_dataset(self, dataset: list) -> int:
        with self._prompt_lock:
            for sample in dataset:
                if isinstance(sample, dict):
                    data_id = sample.get("data_id") or self._generate_data_id()
                    input_ids = sample.get("input_ids")
                    packed_loss_mask = sample.get("packed_loss_mask")
                    if (
                        input_ids is not None
                        and packed_loss_mask is None
                        and not getattr(self.args, "dynamic_loss_mask", False)
                    ):
                        raise ValueError(
                            f"packed_loss_mask is required when input_ids is provided "
                            f"(data_id={data_id}). Enable dynamic_loss_mask or use "
                            f"defer_tokenization=True to match engine-produced token IDs."
                        )
                    metadata = self._work_metadata(
                        data_id=str(data_id),
                        metadata=sample.get("metadata", {}),
                        split="train",
                        epoch=self._dataset_epoch,
                    )
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample.get("prompt", sample),
                        input_ids=input_ids,
                        packed_loss_mask=packed_loss_mask,
                        formatted_prompt=sample.get("formatted_prompt"),
                        metadata=metadata,
                        multimodal_inputs=sample.get("multimodal_inputs"),
                    )
                else:
                    data_id = self._generate_data_id()
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample,
                        metadata=self._work_metadata(
                            data_id=data_id,
                            metadata=None,
                            split="train",
                            epoch=self._dataset_epoch,
                        ),
                    )
                self.prompt_buffer.append(entry)
            return len(dataset)

    def _load_dataset_split(
        self, args, split: str, *, admission_report: dict | None = None
    ) -> list:
        """Load one split from either replay records or conversation data."""
        if split not in ("train", "eval"):
            raise ValueError(f"Unknown dataset split: {split!r}")

        if getattr(args, "inference_engine_type", None) == "offline":
            from aspec.offline.dataset import OfflineDataset

            dataset = OfflineDataset(args.offline_data_path)
            identity = dataset.identity
            rows = [
                {
                    "data_id": str(row["data_id"]),
                    "metadata": {"offline_replay": True, "offline_dataset_identity": identity, "sample_sha256": row["sha256"]},
                    "seq_len": row["seq_len"],
                }
                for row in dataset.rows(split)
            ]
            if admission_report is not None:
                admission_report.update(
                    {
                        "state": "admitted",
                        "source_samples": len(rows),
                        "processed_samples": len(rows),
                        "admitted_samples": len(rows),
                        "skipped_samples": 0,
                        "unbalanced_thinking_samples": 0,
                        "cache_state": "offline_replay",
                    }
                )
            return rows

        data_path = (
            args.train_data_path if split == "train" else getattr(args, "eval_data_path", None)
        )
        if not data_path:
            if admission_report is not None:
                admission_report.update(
                    {
                        "state": "not_configured",
                        "source_samples": 0,
                        "processed_samples": 0,
                        "admitted_samples": 0,
                        "skipped_samples": 0,
                        "unbalanced_thinking_samples": 0,
                        "cache_state": "not_used",
                    }
                )
            return []

        from aspec.data.dataset import load_conversation_dataset

        dataset_args = args
        if split == "eval":
            dataset_args = copy.copy(args)
            dataset_args.train_data_path = data_path
            if getattr(args, "eval_prompt_key", None):
                dataset_args.prompt_key = args.eval_prompt_key
        return load_conversation_dataset(dataset_args, admission_report=admission_report)

    def load_dataset(self, args) -> int:
        """Load and store the training dataset for later epochs."""
        self._dataset_admission = {}
        self._stored_dataset = self._load_dataset_split(
            args, "train", admission_report=self._dataset_admission
        )
        if not self._stored_dataset:
            if getattr(args, "inference_engine_type", None) == "offline":
                raise ValueError(
                    "Offline dataset has no train samples: "
                    f"admission={self._dataset_admission}"
                )
            raise ValueError(
                f"Training dataset is empty after processing. "
                f"Check train_data_path='{args.train_data_path}', "
                f"max_seq_length={getattr(args, 'max_seq_length', None)}, "
                f"and chat_template settings. admission={self._dataset_admission}"
            )
        self._dataset_fingerprint = dataset_fingerprint(self._stored_dataset)
        self._dataset_admission["dataset_fingerprint"] = self._dataset_fingerprint
        self._dataset_admission["admitted_samples"] = len(self._stored_dataset)
        logger.info(f"Controller loaded dataset: {len(self._stored_dataset)} samples")
        return len(self._stored_dataset)

    def _prepare_dataset(
        self,
        skip: int = 0,
        committed_work_ids: set[str] | None = None,
    ) -> list:
        """Return the current epoch in deterministic scheduling order.

        When shuffle is enabled the ordering is deterministic from
        (seed + epoch). Exact resume removes semantic work IDs recorded by the
        checkpoint; numeric ``skip`` remains only for callers that are not
        resuming a committed-work checkpoint.
        """
        data = list(self._stored_dataset)
        if self._shuffle_dataset:
            import random

            rng = random.Random(self._dataset_seed + self._dataset_epoch)
            rng.shuffle(data)

        # Packing needs a mixed candidate window. Pre-sorting each window by
        # length collapses it into same-size runs and defeats the row optimizer.
        if not self.packing_enabled:
            data = length_grouped_order(data, self._length_group_size)

        if skip > 0:
            skip = min(skip, len(data))
            data = data[skip:]

        committed = committed_work_ids or set()
        filtered = 0
        if committed:
            remaining = []
            for sample in data:
                data_id = str(sample.get("data_id") or sample.get("id") or "")
                if not data_id:
                    raise ValueError("Exact resume encountered a row without stable data_id")
                work_id = build_work_identity(
                    split="train", epoch=self._dataset_epoch, source_id=data_id
                )[WORK_ID_KEY]
                if work_id in committed:
                    filtered += 1
                else:
                    remaining.append(sample)
            data = remaining

        shuffle_tag = (
            f"seed {self._dataset_seed}+{self._dataset_epoch}"
            if self._shuffle_dataset
            else "shuffle disabled"
        )
        logger.info(
            f"Prepared dataset ({len(data)} samples, {shuffle_tag}, "
            f"length group {'disabled for packing' if self.packing_enabled else self._length_group_size}"
            + (f", skipped {skip}" if skip > 0 else "")
            + (f", excluded {filtered} committed work items" if filtered else "")
            + ")"
        )
        return data

    def submit_training_dataset(
        self,
        epoch: int = 0,
        skip: int = 0,
        committed_work_ids: list[str] | None = None,
        expected_dataset_fingerprint: str | None = None,
    ) -> int:
        """Submit the stored training dataset to the prompt buffer for inference.

        Args:
            epoch: Current epoch number (for deterministic shuffle seed).
            skip: Legacy deterministic prefix skip for non-resume callers.
            committed_work_ids: Exact semantic work set restored from a checkpoint.
            expected_dataset_fingerprint: Checkpoint dataset identity; mismatch fails closed.
        """
        if self._stored_dataset is None:
            raise RuntimeError("No stored dataset to submit")
        if committed_work_ids is not None and skip:
            raise ValueError("Exact committed-work resume cannot also use numeric skip")
        if expected_dataset_fingerprint is not None:
            if expected_dataset_fingerprint != self._dataset_fingerprint:
                raise ValueError(
                    "Resume dataset fingerprint does not match the checkpoint: "
                    f"current={self._dataset_fingerprint}, "
                    f"checkpoint={expected_dataset_fingerprint}"
                )
            self._restored_committed_work_ids = set(committed_work_ids or [])
        self._dataset_epoch = epoch
        return self.add_dataset(
            self._prepare_dataset(
                skip=skip,
                committed_work_ids=self._restored_committed_work_ids,
            )
        )

    def reload_dataset(self) -> int:
        """Re-add the stored dataset to the prompt buffer (epoch reload)."""
        if self._stored_dataset is None:
            raise RuntimeError("No stored dataset to reload")
        self._dataset_epoch += 1
        return self.add_dataset(
            self._prepare_dataset(
                committed_work_ids=self._restored_committed_work_ids,
            )
        )

    def restart_training_source_after_profile(self) -> dict:
        """Replace unused profile prompts with a fresh training epoch.

        Producer profiling consumes from the same finite source as training.
        Once its output and transfer resources have been released, the
        remaining prompts belong to a partial epoch and cannot provide a
        complete training budget.  Drop only those not-yet-submitted prompts
        and reload the stored dataset under the next epoch identity.
        """
        if self._stored_dataset is None:
            raise RuntimeError("No stored dataset to restart")
        with self._prompt_lock:
            discarded_prompts = len(self.prompt_buffer)
            self.prompt_buffer.clear()
        reloaded_samples = self.reload_dataset()
        return {
            "state": "reloaded",
            "discarded_prompts": discarded_prompts,
            "reloaded_samples": reloaded_samples,
            "dataset_epoch": self._dataset_epoch,
        }

    def get_checkpoint_work_state(self) -> dict:
        """Return the cumulative semantic work set for checkpoint persistence."""

        if self._dataset_fingerprint is None or self._stored_dataset is None:
            raise RuntimeError("cannot checkpoint work state before loading the dataset")
        committed = self._restored_committed_work_ids.union(
            self._lifecycle.committed_work_ids()
        )
        return {
            "dataset_epoch": self._dataset_epoch,
            "dataset_size": len(self._stored_dataset),
            "dataset_fingerprint": self._dataset_fingerprint,
            "dataset_order": {
                "seed": self._dataset_seed,
                "shuffle": self._shuffle_dataset,
                "length_group_size": self._length_group_size,
                "packing": self.packing_enabled,
            },
            "committed_work_ids": sorted(committed),
        }

    def load_eval_dataset(self, args) -> int:
        """Load eval dataset on the controller and store it. Returns size (0 if none)."""
        self._eval_dataset_admission = {}
        raw_dataset = self._load_dataset_split(
            args, "eval", admission_report=self._eval_dataset_admission
        )
        raw_count = len(raw_dataset)
        # Truncate to a multiple of dp_size so every dispatch is a full batch
        usable = (raw_count // self.dp_size) * self.dp_size
        if getattr(args, "eval_data_path", None) and usable == 0:
            raise ValueError(f"Requested eval has no complete DP batch: admitted={raw_count}, dp_size={self.dp_size}")
        if usable < raw_count:
            logger.info(
                f"Eval dataset truncated from {raw_count} to {usable} samples "
                f"(dp_size={self.dp_size})"
            )
        self._stored_eval_dataset = raw_dataset[:usable]
        count = len(self._stored_eval_dataset)
        self._eval_dataset_admission["processed_samples"] = raw_count
        self._eval_dataset_admission["truncated_samples"] = raw_count - usable
        self._eval_dataset_admission["admitted_samples"] = count
        self._eval_dataset_admission["dataset_fingerprint"] = dataset_fingerprint(self._stored_eval_dataset)
        logger.info(f"Controller loaded eval dataset: {count} samples")
        return count

    def get_dataset_admission_evidence(self) -> dict:
        """Return source-to-runtime sample accounting for Evidence."""

        return {
            "train": dict(self._dataset_admission),
            "eval": dict(self._eval_dataset_admission),
        }

    def get_dataset_size(self) -> int:
        if self._stored_dataset is None:
            raise RuntimeError(
                "get_dataset_size() called but no dataset has been loaded. "
                "Call load_dataset() first."
            )
        return len(self._stored_dataset)

    def get_eval_dataset_size(self) -> int:
        return len(self._stored_eval_dataset) if self._stored_eval_dataset is not None else 0

    def compute_vocab_mapping(self, target_vocab_size: int, draft_vocab_size: int) -> tuple:
        """Generate vocab mapping on the controller using the stored dataset.

        Requires the dataset to have been loaded with defer_tokenization=False,
        since vocab mapping needs input_ids.
        """
        from aspec.data.preprocessing import generate_vocab_mapping

        if self._stored_dataset is None:
            raise RuntimeError("No stored dataset for vocab mapping")
        if not self._stored_dataset or "input_ids" not in self._stored_dataset[0]:
            raise ValueError(
                "compute_vocab_mapping requires input_ids in a nonempty dataset. "
                "Set defer_tokenization=False to enable tokenization."
            )
        return generate_vocab_mapping(
            prompts=self._stored_dataset,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

    # ─────────────────────────────────────────────────────────────
    # Interface for Inference Manager
    # ─────────────────────────────────────────────────────────────

    def get_prompts(self, num_prompts: int) -> list[InferenceInput]:
        """Inference manager gets prompts with data_ids.

        Args:
            num_prompts: Maximum number of prompts to fetch.

        Returns:
            List of InferenceInput objects.
        """
        with self._prompt_lock:
            entries = []
            for _ in range(min(num_prompts, len(self.prompt_buffer))):
                entries.append(self.prompt_buffer.popleft())
            return entries

    def push_inference_results(self, results: list[InferenceOutput]) -> int:
        """Inference sends back (data_id, mooncake_key) pairs.

        Controller stores the keys and tracks exact bytes for backpressure.
        Eval results are routed by their scheduler-owned work identity.

        Args:
            results: List of InferenceOutput containing data_id, mooncake_key,
                    tensor_shapes, and tensor_dtypes.

        Returns:
            Current pool bytes after adding results. This allows inference manager
            to implement Mooncake backpressure.
        """
        validated_results = []
        for result in results:
            metadata = dict(result.metadata or {})
            _work_id, _execution_id, work_kind = require_work_identity(metadata)
            if work_kind not in {"train", "eval"}:
                raise ValueError(
                    f"Unsupported {WORK_KIND_KEY}={work_kind!r} for data_id={result.data_id}"
                )
            tracked_as_eval = result.data_id in self._eval_data_ids
            if (work_kind == "eval") != tracked_as_eval:
                raise ValueError(
                    "Inference result work kind disagrees with submitted scheduler state: "
                    f"data_id={result.data_id}, work_kind={work_kind!r}, "
                    f"tracked_as_eval={tracked_as_eval}"
                )
            result.metadata = metadata
            validated_results.append((result, metadata, work_kind))

        eval_results = [
            result for result, _metadata, work_kind in validated_results if work_kind == "eval"
        ]
        train_results = [
            result for result, _metadata, work_kind in validated_results if work_kind == "train"
        ]

        registrations = []
        # Publish ownership and queue entries atomically with owner fencing.
        # Keep the eval -> pool lock order used by retirement and discard.
        with self._eval_pool_lock, self._pool_lock:
            if getattr(self, "_read_owners_fenced", False):
                raise RuntimeError("Cannot admit output after attempt owners were fenced")
            for result, metadata, _work_kind in validated_results:
                sample_bytes = estimate_tensor_bytes(
                    result.tensor_shapes or {},
                    result.tensor_dtypes or {},
                )
                registrations.append(
                    {
                        "data_id": result.data_id,
                        "mooncake_key": result.mooncake_key,
                        "metadata": metadata,
                        "size_bytes": sample_bytes,
                    }
                )
            self._lifecycle.register_outputs(registrations)
            for result, registration in zip(results, registrations, strict=True):
                sample_bytes = registration["size_bytes"]
                self._sample_bytes[result.mooncake_key] = sample_bytes
            self._pool_bytes = self._lifecycle.resident_bytes

            self.eval_pool.extend(eval_results)
            self.sample_pool.extend(train_results)
            for result in train_results:
                length = self._seq_len(result)
                self._observed_sequence_count += 1
                self._observed_sequence_max = max(self._observed_sequence_max, length)
                self._observed_sequence_min = (
                    length
                    if self._observed_sequence_min is None
                    else min(self._observed_sequence_min, length)
                )

        pool_bytes = self._lifecycle.resident_bytes

        self._inference_monitor.record(len(results))
        return pool_bytes

    def get_prompt_buffer_size(self) -> int:
        """Get current size of prompt buffer."""
        return len(self.prompt_buffer)

    # ─────────────────────────────────────────────────────────────
    # Interface for Training
    # ─────────────────────────────────────────────────────────────

    def get_train_queues(self) -> list[Queue]:
        """Get the per-DP training queues."""
        return self.train_queues

    def get_ready_pool_size(self) -> int:
        """Return completed outputs still waiting for controller dispatch.

        This is the host-orchestrator queue occupancy used for inference
        admission.  Outputs already dispatched to a trainer remain visible in
        the ownership ledger until their consumer releases them, but must not
        continue consuming a scheduler queue credit.
        """

        with self._pool_lock:
            train_size = len(self.sample_pool)
        with self._eval_pool_lock:
            eval_size = len(self.eval_pool)
        return train_size + eval_size

    def get_ready_pool_size_by_node(self) -> dict[str, int]:
        """Return node-local ready occupancy without folding locality credits."""

        if self.feature_affinity != "node_local_required":
            return {}
        with self._pool_lock:
            train = list(self.sample_pool)
        with self._eval_pool_lock:
            ready = train + list(self.eval_pool)

        counts = {node: 0 for node in self._dp_ranks_by_node}
        for sample in ready:
            metadata = sample.metadata or {}
            if metadata.get("feature_affinity") != "node_local_required":
                raise RuntimeError(
                    "node-local ready pool contains output without the required contract"
                )
            node = str(metadata.get("source_node_ip", ""))
            if node not in counts:
                raise RuntimeError(
                    f"node-local ready pool contains unplanned source node {node!r}"
                )
            counts[node] += 1
        return counts

    def get_pool_size(self) -> int:
        """Return outputs whose transferred storage remains lifecycle-resident.

        This is an ownership/storage observation, not host-orchestrator queue
        occupancy.  In particular, dispatched outputs remain resident until
        every expected consumer reports release.
        """
        with self._pool_lock:
            return self._lifecycle.resident_outputs

    def get_pool_bytes(self) -> int:
        """Get current bytes in sample pool (for Mooncake backpressure)."""
        with self._pool_lock:
            return self._lifecycle.resident_bytes

    def release_inference_output(
        self,
        *,
        execution_id: str,
        data_id: str,
        mooncake_key: str,
        participant: str,
        outcome: str,
    ) -> dict:
        """A consumer reports that its transferred storage is no longer owned."""

        with self._pool_lock:
            before = self._lifecycle.resident_outputs
            self._lifecycle.release_output(
                execution_id=execution_id,
                data_id=data_id,
                mooncake_key=mooncake_key,
                participant=participant,
                outcome=outcome,
            )
            if self._lifecycle.resident_outputs < before:
                self._sample_bytes.pop(mooncake_key, None)
            self._pool_bytes = self._lifecycle.resident_bytes
            return {
                "resident_outputs": self._lifecycle.resident_outputs,
                "resident_bytes": self._lifecycle.resident_bytes,
            }

    def commit_optimizer_step(
        self,
        optimizer_step: int,
        execution_ids: list[str],
        expected_unique: int,
    ) -> dict:
        """Commit exactly the work consumed by one successful optimizer step."""

        with self._pool_lock:
            self._lifecycle.commit_optimizer_step(
                optimizer_step=optimizer_step,
                execution_ids=execution_ids,
                expected_unique=expected_unique,
                expected_occurrences=self.sp_size,
            )
            return self._lifecycle.status()

    def pop_unused_inference_results(self, limit: int = 32) -> list[InferenceOutput]:
        """Claim unassigned outputs so shutdown can release their storage."""

        outputs = self.pop_unused_training_results(limit)
        if len(outputs) >= limit:
            return outputs
        with self._eval_pool_lock, self._pool_lock:
            while self.eval_pool and len(outputs) < limit:
                output = self.eval_pool.popleft()
                self._lifecycle.begin_discard(output.metadata)
                outputs.append(output)
        return outputs

    def pop_unused_training_results(self, limit: int = 32) -> list[InferenceOutput]:
        """Claim only unassigned training outputs for an explicit sink."""

        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        outputs: list[InferenceOutput] = []
        with self._pool_lock:
            while self.sample_pool and len(outputs) < limit:
                output = self.sample_pool.popleft()
                self._lifecycle.begin_discard(output.metadata)
                outputs.append(output)
        return outputs

    def confirm_discarded_inference_results(self, outputs: list[InferenceOutput]) -> dict:
        """Complete ownership transfer after unused output storage was removed."""

        with self._pool_lock:
            self._lifecycle.confirm_discarded([output.metadata for output in outputs])
            for output in outputs:
                self._sample_bytes.pop(output.mooncake_key, None)
            self._pool_bytes = self._lifecycle.resident_bytes
            return self._lifecycle.status()

    def get_work_lifecycle_status(self) -> dict:
        with self._pool_lock:
            return self._lifecycle.status()

    def finalize_fenced_work(self, *, owners_fenced: bool) -> dict:
        """Retire old-attempt outputs only after the driver proves owner fencing."""
        if not owners_fenced:
            raise RuntimeError("Read owners must be fenced before retiring outputs")
        with self._eval_pool_lock, self._pool_lock:
            self._read_owners_fenced = True
            discarded = self._lifecycle.discard_all_after_owner_fence()
            self.sample_pool.clear()
            self.eval_pool.clear()
            self._sample_bytes.clear()
            self._pool_bytes = self._lifecycle.resident_bytes
            return {"discarded": discarded, "lifecycle": self._lifecycle.status()}

    def finalize_uncommitted_work(self) -> dict:
        """Mark storage-free, uncommitted training work discarded on shutdown."""

        with self._pool_lock:
            discarded = self._lifecycle.discard_released_uncommitted()
            return {
                "discarded": discarded,
                "lifecycle": self._lifecycle.status(),
            }

    # ─────────────────────────────────────────────────────────────
    # Dispatch Logic
    # ─────────────────────────────────────────────────────────────

    def set_inference_error(self, msg: str) -> None:
        """Called by the inference manager when a fatal error occurs."""
        self._inference_error = msg

    def try_dispatch_batch(
        self,
        optimizer_step: int | None = None,
        microstep: int | None = None,
        force_packing_progress: bool = False,
        final_source_drained: bool = False,
        max_packing_documents: int | None = None,
        return_document_count: bool = False,
    ) -> bool | int:
        """Try to dispatch one batch to training queues.

        Only dispatches when sample pool has enough samples (>= dispatch_batch_size).
        Dispatches TrainSample objects that MooncakeDataFetcher can consume.
        Subtracts bytes from pool tracking when dispatching.

        Returns:
            True if a batch was dispatched, False if not enough samples.  A
            finite packing caller may request the dispatched document count.

        Raises:
            RuntimeError: If the inference manager has reported a fatal error.
        """
        if self._inference_error is not None:
            raise RuntimeError(f"Inference engine failed: {self._inference_error}")
        if self.packing_enabled:
            return self._try_dispatch_packed(
                optimizer_step=optimizer_step,
                microstep=microstep,
                force_progress=force_packing_progress,
                final_source_drained=final_source_drained,
                max_documents=max_packing_documents,
                return_document_count=return_document_count,
            )

        with self._pool_lock:
            pool_size = len(self.sample_pool)
            now = time.time()
            should_log = (now - self._last_dispatch_log_time) >= 2.0
            if pool_size < self.dispatch_batch_size:
                if should_log:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"try_dispatch_batch: pool_size={pool_size} < dispatch_batch_size={self.dispatch_batch_size}, not dispatching"
                    )
                return False

            if should_log:
                self._last_dispatch_log_time = now
                logger.debug(
                    f"try_dispatch_batch: pool_size={pool_size} >= dispatch_batch_size={self.dispatch_batch_size}, dispatching batch {self.batch_id}"
                )

            if getattr(self, "feature_affinity", "any") == "node_local_required":
                required_by_node = {
                    node: len(ranks) * int(self.args.per_dp_rank_batch_size)
                    for node, ranks in self._dp_ranks_by_node.items()
                }
                batch_results = take_node_local_balanced_batch(
                    self.sample_pool, required_by_node
                )
                if batch_results is None:
                    return False
            elif self._source_balanced_dispatch:
                batch_results = take_source_aware_batch(
                    self.sample_pool, self.dispatch_batch_size
                )
                if batch_results is None:
                    return False
            else:
                batch_results = [
                    self.sample_pool.popleft()
                    for _ in range(self.dispatch_batch_size)
                ]

        with self._pool_lock:
            for result in batch_results:
                self._lifecycle.mark_dispatched(
                    result.metadata,
                    expected_consumers=self.sp_size,
                )
        self._dispatch_to_queues(
            batch_results,
            self.train_queues,
            optimizer_step=optimizer_step,
            microstep=microstep,
        )

        self._training_monitor.record(self.dispatch_batch_size)
        logger.debug(
            f"Dispatched batch {self.batch_id} with {self.dispatch_batch_size} samples "
            f"to {self.dp_size} queues at t={time.time():.3f}"
        )
        self.batch_id += 1
        return True

    def _try_dispatch_packed(
        self,
        optimizer_step: int | None = None,
        microstep: int | None = None,
        force_progress: bool = False,
        final_source_drained: bool = False,
        max_documents: int | None = None,
        return_document_count: bool = False,
    ) -> bool | int:
        """Dispatch one controller-owned packed row to every DP rank."""
        with self._pool_lock:
            pool_size = len(self.sample_pool)
            pool_capacity = int(getattr(self.args, "max_sample_pool_size", 0) or 0)
            force_progress = force_progress or (
                pool_capacity > 0 and pool_size >= pool_capacity
            )
            if pool_size < self.dp_size or (
                pool_size < self.packing_min_samples_per_dispatch and not force_progress
            ):
                return False

            samples = list(self.sample_pool)
            lengths = [self._seq_len(sample) for sample in samples]
            useful_work: list[int] = []
            for sample, length in zip(samples, lengths, strict=True):
                if sample.packed_loss_mask is None:
                    raise ValueError(
                        "dflash_packing requires packed_loss_mask for work-aware "
                        f"scheduling; missing for data_id={sample.data_id}"
                    )
                if getattr(self.args, "draft_algorithm", None) in {"eagle3", "eagle4"}:
                    useful_work.append(count_eagle_supervised_positions(
                        sample.packed_loss_mask, length, int(self.args.ttt_length)
                    ))
                else:
                    useful_work.append(
                        count_eligible_anchor_starts(
                            sample.packed_loss_mask, length,
                            int(getattr(self.args, "dflash_block_size", 16)),
                        )
                    )
            if getattr(self, "feature_affinity", "any") == "node_local_required":
                for sample in samples:
                    metadata = sample.metadata or {}
                    source_node = str(metadata.get("source_node_ip", ""))
                    if (
                        metadata.get("feature_affinity") != "node_local_required"
                        or source_node not in self._dp_ranks_by_node
                    ):
                        raise ValueError(
                            "node-local packing received output without a planned "
                            f"source-node contract: data_id={sample.data_id!r}, "
                            f"source_node_ip={source_node!r}"
                        )
                rank_rows: list[list[int]] = [[] for _ in range(self.dp_size)]
                selected_indices: list[int] = []
                packed_tokens = 0
                capacity_tokens = 0
                useful_anchor_starts = 0
                node_items = list(self._dp_ranks_by_node.items())
                extra_capacity = (
                    max_documents - self.dp_size
                    if max_documents is not None
                    else None
                )
                node_extras = {node: 0 for node, _ in node_items}
                if extra_capacity is not None:
                    weighted = [
                        (extra_capacity * len(ranks) // self.dp_size, node)
                        for node, ranks in node_items
                    ]
                    for share, node in weighted:
                        node_extras[node] = share
                    remainder = extra_capacity - sum(node_extras.values())
                    for _, node in sorted(
                        (
                            (-(extra_capacity * len(ranks) % self.dp_size), node)
                            for node, ranks in node_items
                        )
                    )[:remainder]:
                        node_extras[node] += 1
                for node, ranks in node_items:
                    node_indices = [
                        index
                        for index, sample in enumerate(samples)
                        if str((sample.metadata or {}).get("source_node_ip", "")) == node
                    ]
                    local_lengths = [lengths[index] for index in node_indices]
                    local_work = [useful_work[index] for index in node_indices]
                    local = pack_work_aware_rows(
                        local_lengths,
                        local_work,
                        dp_size=len(ranks),
                        max_seq_length=self.packing_max_seq,
                        max_selected=(
                            len(ranks) + node_extras[node]
                            if extra_capacity is not None
                            else None
                        ),
                    )
                    if local is None:
                        return False
                    local_remainder = len(node_indices) - len(local.selected_indices)
                    if (
                        final_source_drained
                        and 0 < local_remainder < len(ranks)
                        and len(node_indices) >= 2 * len(ranks)
                    ):
                        local = pack_work_aware_rows(
                            local_lengths,
                            local_work,
                            dp_size=len(ranks),
                            max_seq_length=self.packing_max_seq,
                            max_selected=len(node_indices) - len(ranks),
                        )
                        if local is None:
                            raise RuntimeError(
                                "Failed to reserve a final node-local packing dispatch"
                            )
                    for rank, local_row in zip(ranks, local.rank_rows, strict=True):
                        rank_rows[rank] = [node_indices[index] for index in local_row]
                    selected_indices.extend(
                        node_indices[index] for index in local.selected_indices
                    )
                    packed_tokens += local.packed_tokens
                    capacity_tokens += local.capacity_tokens
                    useful_anchor_starts += local.useful_anchor_starts
                packed = PackingResult(
                    rank_rows=rank_rows,
                    selected_indices=tuple(sorted(selected_indices)),
                    packed_tokens=packed_tokens,
                    capacity_tokens=capacity_tokens,
                    useful_anchor_starts=useful_anchor_starts,
                )
            else:
                packed = pack_work_aware_rows(
                    lengths,
                    useful_work,
                    dp_size=self.dp_size,
                    max_seq_length=self.packing_max_seq,
                    max_selected=max_documents,
                )
            if packed is None:
                return False
            remainder = pool_size - len(packed.selected_indices)
            if (
                getattr(self, "feature_affinity", "any") != "node_local_required"
                and final_source_drained
                and 0 < remainder < self.dp_size
                and pool_size >= 2 * self.dp_size
            ):
                # The producer is exhausted, so a stranded sub-DP remainder can
                # never be refilled.  Reserve one complete final dispatch rather
                # than consuming almost the entire pool in this packed row set.
                packed = pack_work_aware_rows(
                    lengths,
                    useful_work,
                    dp_size=self.dp_size,
                    max_seq_length=self.packing_max_seq,
                    max_selected=pool_size - self.dp_size,
                )
                if packed is None:
                    raise RuntimeError("Failed to reserve a final packing dispatch")

            selected_set = set(packed.selected_indices)
            selected_samples = {index: samples[index] for index in packed.selected_indices}
            self.sample_pool.clear()
            self.sample_pool.extend(
                sample for index, sample in enumerate(samples) if index not in selected_set
            )

            for sample in selected_samples.values():
                self._lifecycle.mark_dispatched(sample.metadata, expected_consumers=self.sp_size)
                self._sample_bytes.pop(sample.mooncake_key, None)
            self._pool_bytes = self._lifecycle.resident_bytes

        for dp_rank, row in enumerate(packed.rank_rows):
            row_id = self._row_id_counter
            self._row_id_counter += 1
            row_tokens = sum(
                self._seq_len(selected_samples[index]) for index in row
            )
            self._packing_observed_row_count += 1
            self._packing_observed_row_max = max(
                self._packing_observed_row_max, row_tokens
            )
            self._packing_observed_row_min = (
                row_tokens
                if self._packing_observed_row_min is None
                else min(self._packing_observed_row_min, row_tokens)
            )
            self._packing_max_documents_per_row = max(
                self._packing_max_documents_per_row, len(row)
            )
            for position, index in enumerate(row):
                result = selected_samples[index]
                metadata = dict(getattr(result, "metadata", {}) or {})
                metadata = self._stamp_feature_affinity(
                    metadata, dp_rank=dp_rank, count_dispatch=True
                )
                if optimizer_step is not None:
                    metadata["optimizer_step"] = optimizer_step
                if microstep is not None:
                    metadata["microstep"] = microstep
                sample = TrainSample(
                        mooncake_key=result.mooncake_key,
                        tensor_shapes=result.tensor_shapes,
                        tensor_dtypes=result.tensor_dtypes,
                        packed_loss_mask=result.packed_loss_mask,
                        last_turn_loss_only=metadata.get("has_thinking"),
                        metadata=metadata,
                        data_id=result.data_id,
                        row_id=row_id,
                        last_in_row=position == len(row) - 1,
                )
                start = dp_rank * self.sp_size
                for rank in range(start, start + self.sp_size):
                    self.train_queues[rank].put(sample)

        documents = len(packed.selected_indices)
        fill_ratio = packed.fill_ratio
        self._packing_rows_dispatched += self.dp_size
        self._packing_documents_dispatched += documents
        self._packing_tokens_dispatched += packed.packed_tokens
        self._packing_capacity_tokens += packed.capacity_tokens
        self._packing_useful_anchor_starts += packed.useful_anchor_starts
        self._packing_min_fill_ratio = (
            fill_ratio
            if self._packing_min_fill_ratio is None
            else min(self._packing_min_fill_ratio, fill_ratio)
        )
        self._training_monitor.record(documents)
        self.batch_id += 1
        logger.debug(
            "Dispatched packed batch %s: %s documents -> %s rows, fill=%.1f%%",
            self.batch_id - 1,
            documents,
            self.dp_size,
            fill_ratio * 100,
        )
        return documents if return_document_count else True

    def get_scheduler_evidence(self) -> dict:
        """Return scheduler facts separately from generic numeric status."""
        capacity = self._packing_capacity_tokens
        return {
            "packing_enabled": self.packing_enabled,
            "batching_policy": (
                "packed_rows" if self.packing_enabled else "length_grouped_dispatch"
                if self.args.per_dp_rank_batch_size > 1 else "single_document"
            ),
            "batching_window_documents": (
                self.dispatch_batch_size if not self.packing_enabled else None
            ),
            "feature_affinity": getattr(self, "feature_affinity", "any"),
            "node_local_dispatch_samples": getattr(
                self, "_node_local_dispatch_samples", 0
            ),
            "cross_node_dispatch_samples": getattr(
                self, "_cross_node_dispatch_samples", 0
            ),
            "node_local_dispatch_by_node": dict(
                getattr(self, "_node_local_dispatch_by_node", {})
            ),
            "rows_dispatched": self._packing_rows_dispatched,
            "documents_dispatched": self._packing_documents_dispatched,
            "packed_tokens": self._packing_tokens_dispatched,
            "capacity_tokens": capacity,
            "average_fill_ratio": (
                self._packing_tokens_dispatched / capacity if capacity else 0.0
            ),
            "minimum_dispatch_fill_ratio": self._packing_min_fill_ratio or 0.0,
            "maximum_documents_per_row": self._packing_max_documents_per_row,
            "observed_packed_row_count": self._packing_observed_row_count,
            "observed_packed_row_min": self._packing_observed_row_min or 0,
            "observed_packed_row_max": self._packing_observed_row_max,
            ("useful_training_positions" if getattr(self.args, "draft_algorithm", None) in {"eagle3", "eagle4"}
             else "useful_anchor_starts"): self._packing_useful_anchor_starts,
            "observed_sequence_count": self._observed_sequence_count,
            "observed_sequence_min": self._observed_sequence_min or 0,
            "observed_sequence_max": self._observed_sequence_max,
        }

    @staticmethod
    def _seq_len(result: InferenceOutput) -> int:
        shapes = result.tensor_shapes or {}
        ids_shape = shapes.get("input_ids")
        return ids_shape[-1] if ids_shape else 0

    def _partition_results(
        self, results: list[InferenceOutput], *, rank_loads: list[int] | None = None
    ) -> list[list[InferenceOutput]]:
        """Partition InferenceOutputs across DP ranks.

        ASpec forms complete, length-adjacent microbatches before balancing
        their padded context loads. Source selection has already happened;
        locality domains, dispatch cardinality and work identity stay intact.
        Optional loads balance successive dispatches of one optimizer step.
        MBS1 and irregular/eval batches retain their round-robin assignment.
        """
        if getattr(self, "feature_affinity", "any") == "node_local_required":
            partitions: list[list[InferenceOutput]] = [[] for _ in range(self.dp_size)]
            for node, ranks in self._dp_ranks_by_node.items():
                local_results = [
                    result
                    for result in results
                    if str((result.metadata or {}).get("source_node_ip", "")) == node
                ]
                if not local_results or len(local_results) % len(ranks):
                    raise RuntimeError(
                        "node-local feature batch cannot be divided across local Draft "
                        f"ranks: node={node}, samples={len(local_results)}, ranks={len(ranks)}"
                    )
                local_partitions = self._partition_result_subset(
                    local_results, len(ranks),
                    rank_loads=([rank_loads[rank] for rank in ranks]
                                if rank_loads is not None else None),
                )
                for rank, local_partition in zip(ranks, local_partitions, strict=True):
                    partitions[rank] = local_partition
            if sum(len(partition) for partition in partitions) != len(results):
                raise RuntimeError(
                    "node-local feature batch contains missing or unplanned source nodes"
                )
        else:
            partitions = self._partition_result_subset(
                results, self.dp_size, rank_loads=rank_loads
            )
        if rank_loads is not None:
            for rank, partition in enumerate(partitions):
                rank_loads[rank] += self._padded_context_tokens(partition)
        return partitions

    def _padded_context_tokens(self, results: list[InferenceOutput]) -> int:
        """Metadata-only BSHD shape count; CP replicas are counted once."""
        length = max((self._seq_len(result) for result in results), default=0)
        return len(results) * length

    def _partition_result_subset(
        self, results: list[InferenceOutput], rank_count: int,
        *, rank_loads: list[int] | None = None,
    ) -> list[list[InferenceOutput]]:
        """Group similar lengths, then assign whole batches in one locality domain.

        Balancing individual documents by length pairs long and short rows,
        increasing padding. Balance the resulting rectangular executions instead.
        This uses the already selected dispatch only: no extra ready-pool wait,
        reservation, source-ring retention or sorting of an entire global batch.
        """

        partitions: list[list[InferenceOutput]] = [[] for _ in range(rank_count)]
        if rank_count <= 1 or len(results) <= rank_count or len(results) % rank_count != 0:
            for i, result in enumerate(results):
                partitions[i % rank_count].append(result)
            return partitions

        capacity = len(results) // rank_count
        ordered = sorted(results, key=self._seq_len, reverse=True)
        loads = rank_loads if rank_loads is not None else [0] * rank_count
        for start in range(0, len(ordered), capacity):
            min_rank = min(
                (r for r in range(rank_count) if not partitions[r]),
                key=lambda r: loads[r],
            )
            partitions[min_rank] = ordered[start:start + capacity]
        return partitions

    def _stamp_feature_affinity(
        self, metadata: dict, *, dp_rank: int, count_dispatch: bool
    ) -> dict:
        if getattr(self, "feature_affinity", "any") != "node_local_required":
            return metadata
        source_node = str(metadata.get("source_node_ip", ""))
        assigned_node = self._dp_node_by_rank[dp_rank]
        if metadata.get("feature_affinity") != "node_local_required":
            raise RuntimeError(
                "node-local Draft dispatch received output without the required "
                "feature-affinity contract"
            )
        if source_node != assigned_node:
            self._cross_node_dispatch_samples += 1
            raise RuntimeError(
                "node-local feature affinity violation: "
                f"source={source_node!r}, assigned={assigned_node!r}, dp_rank={dp_rank}"
            )
        metadata["assigned_draft_node_ip"] = assigned_node
        if count_dispatch:
            self._node_local_dispatch_samples += 1
            self._node_local_dispatch_by_node[assigned_node] += 1
        return metadata

    def _dispatch_to_queues(
        self,
        batch_results: list[InferenceOutput],
        queues: list[Queue],
        *,
        optimizer_step: int | None = None,
        microstep: int | None = None,
    ) -> None:
        """Partition results across DP ranks and push TrainSamples into queues."""
        training = queues is self.train_queues and not self.packing_enabled
        rank_loads = None
        if training and optimizer_step is not None:
            if self._partition_step != optimizer_step:
                self._partition_step = optimizer_step
                self._partition_rank_loads = [0] * self.dp_size
            rank_loads = self._partition_rank_loads
        partitioned = self._partition_results(batch_results, rank_loads=rank_loads)
        for dp_rank, results in enumerate(partitioned):
            for result in results:
                metadata = dict(getattr(result, "metadata", {}) or {})
                metadata = self._stamp_feature_affinity(
                    metadata,
                    dp_rank=dp_rank,
                    count_dispatch=queues is self.train_queues,
                )
                if optimizer_step is not None:
                    metadata["optimizer_step"] = optimizer_step
                if microstep is not None:
                    metadata["microstep"] = microstep
                last_turn_loss_only = metadata.get("has_thinking")
                sample = TrainSample(
                    mooncake_key=result.mooncake_key,
                    tensor_shapes=result.tensor_shapes,
                    tensor_dtypes=result.tensor_dtypes,
                    packed_loss_mask=result.packed_loss_mask,
                    last_turn_loss_only=last_turn_loss_only,
                    metadata=metadata,
                    data_id=result.data_id,
                )
                if self.sp_size > 1 and len(queues) == self.queue_count:
                    start = dp_rank * self.sp_size
                    for rank in range(start, start + self.sp_size):
                        queues[rank].put(sample)
                else:
                    queues[dp_rank].put(sample)
        if training:
            with self._pool_lock:
                self._batching_input_tokens += sum(self._seq_len(result) for result in batch_results)
                self._batching_context_tokens += sum(
                    self._padded_context_tokens(results) for results in partitioned
                )
                self._batching_microbatches += len(partitioned)

    def push_inference_sample(self, sample: InferenceOutput) -> int:
        """Add a single inference sample to the training pool.

        This method is used by OnlineServingController to add samples
        captured from the inference engine's /generate API.

        Args:
            sample: InferenceOutput containing mooncake_key and tensor metadata.

        Returns:
            Current pool bytes after adding the sample.
        """
        return self.push_inference_results([sample])

    # ─────────────────────────────────────────────────────────────
    # Eval Pipeline
    # ─────────────────────────────────────────────────────────────

    def _build_eval_entries(self, dataset: list) -> list[InferenceInput]:
        eval_entries: list[InferenceInput] = []
        for sample in dataset:
            if isinstance(sample, dict):
                raw_id = sample.get("data_id") or self._generate_data_id()
                data_id = (
                    str(raw_id)
                    if getattr(self.args, "inference_engine_type", None) == "offline"
                    else f"eval_{raw_id}"
                )
                self._eval_data_ids.add(data_id)
                metadata = self._work_metadata(
                    data_id=data_id,
                    metadata=sample.get("metadata", {}),
                    split="eval",
                    epoch=0,
                )
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample.get("prompt", sample),
                    input_ids=sample.get("input_ids"),
                    packed_loss_mask=sample.get("packed_loss_mask"),
                    formatted_prompt=sample.get("formatted_prompt"),
                    metadata=metadata,
                    multimodal_inputs=sample.get("multimodal_inputs"),
                )
            else:
                data_id = f"eval_{self._generate_data_id()}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample,
                    metadata=self._work_metadata(
                        data_id=data_id,
                        metadata=None,
                        split="eval",
                        epoch=0,
                    ),
                )
            eval_entries.append(entry)
        return eval_entries

    def submit_eval_chunk(self, start: int, end: int) -> int:
        """Submit a slice of the stored eval dataset for inference."""
        if self._stored_eval_dataset is None:
            raise RuntimeError("No stored eval dataset")
        chunk = self._stored_eval_dataset[start:end]
        if not chunk:
            return 0

        if start == 0:
            self._eval_expected_count = len(self._stored_eval_dataset)
            self._eval_dispatched_samples = 0

        eval_entries = self._build_eval_entries(chunk)

        with self._prompt_lock:
            self.prompt_buffer.extendleft(reversed(eval_entries))
        logger.info(
            f"Eval: submitted chunk [{start}:{end}] "
            f"({len(chunk)} samples, total_expected={self._eval_expected_count})"
        )
        return len(chunk)

    def get_eval_pool_size(self) -> int:
        with self._eval_pool_lock:
            return len(self.eval_pool)

    def get_eval_queues(self) -> list[Queue]:
        return self.eval_queues

    def try_dispatch_eval_batch(self) -> bool:
        """Dispatch one eval batch from the pool if enough samples are available."""
        bs = self.eval_dispatch_batch_size
        with self._eval_pool_lock:
            if len(self.eval_pool) < bs:
                return False
            if getattr(self, "feature_affinity", "any") == "node_local_required":
                required_by_node = {
                    node: len(ranks) for node, ranks in self._dp_ranks_by_node.items()
                }
                batch_results = take_node_local_balanced_batch(
                    self.eval_pool, required_by_node
                )
                if batch_results is None:
                    return False
            elif self._source_balanced_dispatch:
                batch_results = take_source_aware_batch(self.eval_pool, bs)
                if batch_results is None:
                    return False
            else:
                batch_results = [self.eval_pool.popleft() for _ in range(bs)]

        with self._pool_lock:
            for result in batch_results:
                self._lifecycle.mark_dispatched(
                    result.metadata,
                    expected_consumers=self.sp_size,
                )
        self._dispatch_to_queues(batch_results, self.eval_queues)
        self._eval_dispatched_samples += bs
        logger.debug(
            f"Eval: dispatched batch ({self._eval_dispatched_samples}/"
            f"{self._eval_expected_count} samples)"
        )
        return True

    def finalize_eval_dispatch(self) -> None:
        """Verify all eval batches were dispatched, then clean up tracking state."""
        with self._eval_pool_lock:
            arrived = self._eval_dispatched_samples + len(self.eval_pool)
            pool_remaining = len(self.eval_pool)

        if self._eval_expected_count <= 0 or arrived < self._eval_expected_count:
            raise RuntimeError(
                "finalize_eval_dispatch called before all samples arrived "
                f"(arrived={arrived}, expected={self._eval_expected_count})"
            )
        if pool_remaining >= self.eval_dispatch_batch_size:
            raise RuntimeError(
                "finalize_eval_dispatch called with undispatched full batches "
                f"(pool={pool_remaining}, batch_size={self.eval_dispatch_batch_size})"
            )

        with self._eval_pool_lock:
            if pool_remaining > 0:
                logger.info(
                    f"Eval: dropping {pool_remaining} leftover samples that didn't fill a batch"
                )
                self.eval_pool.clear()

        self._eval_data_ids.clear()
        self._eval_expected_count = 0
        self._eval_dispatched_samples = 0
        logger.info("Eval: dispatch finalized, tracking state cleared")

    # ─────────────────────────────────────────────────────────────
    # Status and Monitoring
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current status of controller."""
        # Shared-source dispatch can mutate this pool concurrently with a
        # monitoring RPC. Snapshot the ledger and queue under the same lock.
        with self._pool_lock:
            status = {
                "prompt_buffer_size": len(self.prompt_buffer),
                "sample_pool_size": len(self.sample_pool),
                "resident_output_size": self._lifecycle.resident_outputs,
                "resident_output_bytes": self._lifecycle.resident_bytes,
                "batches_dispatched": self.batch_id,
                "dispatch_batch_size": self.dispatch_batch_size,
                # Cumulative dispatched shapes, not committed throughput or a
                # trainer wall-time estimate. Excludes eval and the THD packer.
                "batching_dispatched_input_tokens": self._batching_input_tokens,
                "batching_dispatched_context_tokens": self._batching_context_tokens,
                "batching_dispatched_microbatches": self._batching_microbatches,
                "batching_context_efficiency": (
                    self._batching_input_tokens / self._batching_context_tokens
                    if self._batching_context_tokens else 0.0
                ),
            }
            if self.feature_affinity == "node_local_required":
                status["ready_pool_by_source_node"] = dict(
                    sorted(
                        Counter(
                            str((sample.metadata or {}).get("source_node_ip", "<missing>"))
                            for sample in self.sample_pool
                        ).items()
                    )
                )
                status["dispatch_samples_required_by_node"] = {
                    node: len(ranks) * int(self.args.per_dp_rank_batch_size)
                    for node, ranks in self._dp_ranks_by_node.items()
                }
            return status

    def get_speeds(self) -> dict:
        """Get current throughput speeds in entries/sec."""
        elapsed = time.time() - self._start_time
        return {
            "inference_speed": round(self._inference_monitor.get_speed(), 2),
            "training_speed": round(self._training_monitor.get_speed(), 2),
            "inference_total": self._inference_monitor.get_total_count(),
            "training_total": self._training_monitor.get_total_count(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_inference_speed": round(
                self._inference_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
            "avg_training_speed": round(
                self._training_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
        }

    def get_full_status(self) -> dict:
        """Get complete status including speeds."""
        status = self.get_status()
        status.update(self.get_speeds())
        return status

    def shutdown(self) -> None:
        """Signal training workers to stop by sending None to queues."""
        if self._shutdown_signaled:
            return
        self._shutdown_signaled = True
        for q in self.train_queues:
            q.put(None)
        logger.info("Controller shutdown: sent stop signals to training queues")


# Keep the existing Ray interface while allowing the shared-source coordinator
# to reuse the same controller state machine for independent experiments.
TrainingControllerState = AsyncTrainingController
AsyncTrainingController = ray.remote(TrainingControllerState)
