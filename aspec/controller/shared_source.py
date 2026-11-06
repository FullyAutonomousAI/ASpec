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

"""One source stream, with independent retained training-controller states."""

from __future__ import annotations

import copy
import json
import socket
import threading
from collections import deque

import ray

from aspec.config.draft_execution import is_direct_cp_backend
from aspec.controller.training_controller import TrainingControllerState
from aspec.transfer.mooncake.p2p import (
    MooncakeP2PDescriptor,
    MooncakeP2PTransfer,
    pack_direct_cp_p2p_keys,
    shared_source_cp_views,
)
from aspec.work_lifecycle import EXECUTION_ID_KEY, WORK_ID_KEY


def source_control(descriptor: MooncakeP2PDescriptor, **operation) -> dict:
    """Generation-checked control; retries cannot change a frozen audience."""

    if not descriptor.source_release_endpoint:
        raise RuntimeError("shared source requires an acknowledged control endpoint")
    host, port = descriptor.source_release_endpoint.rsplit(":", 1)
    message = {
        "run_id": descriptor.run_id,
        "attempt_id": descriptor.attempt_id,
        "slot_id": descriptor.slot_id,
        "generation": descriptor.generation,
        "store_key": descriptor.store_key,
        "payload_bytes": descriptor.payload_bytes,
        **operation,
    }
    last_error = None
    for _attempt in range(3):
        try:
            with socket.create_connection((host, int(port)), timeout=5) as connection:
                connection.settimeout(5)
                connection.sendall(json.dumps(message).encode() + b"\n")
                response = MooncakeP2PTransfer._recv_release_message(connection)
            if response.get("ok") is not True:
                raise RuntimeError(f"shared source rejected control: {response.get('error')}")
            return response
        except (OSError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError("shared source control was not acknowledged") from last_error


class _ExperimentMethod:
    def __init__(self, hub, name: str, method: str):
        self.hub, self.name, self.method = hub, name, method

    def remote(self, *args, **kwargs):
        return self.hub.invoke.remote(self.name, self.method, args, kwargs)


class ExperimentController:
    """Use the unchanged trainer/loop RPC interface with an experiment namespace."""

    def __init__(self, hub, name: str):
        self.hub, self.name = hub, name

    def __getattr__(self, method: str):
        if method.startswith("_"):
            raise AttributeError(method)
        return _ExperimentMethod(self.hub, self.name, method)


class SharedSourceState:
    def __init__(self, source_args, experiment_args: dict):
        self.source = TrainingControllerState(
            source_args, 1, with_train_queues=False, with_eval_queues=False
        )
        self.experiments = {
            name: TrainingControllerState(args, args.dp_size, with_eval_queues=bool(getattr(args, "eval_data_path", None)))
            for name, args in experiment_args.items()
        }
        # Independent optimizers may use different RNG seeds; source scheduling
        # belongs to the common stream rather than either training RNG.
        for child in self.experiments.values():
            child._dataset_seed = self.source._dataset_seed
            child._source_balanced_dispatch = False
        self._reader_payloads = {}
        self.remaining = {
            name: (None if getattr(args, "dflash_packing", False) and args.num_train_steps else (args.num_train_steps or 0) * args.global_batch_size)
            for name, args in experiment_args.items()
        }
        self.active = set(self.experiments)
        self._retired = set()
        self.ready: set[str] = set()
        self._lock = threading.RLock()
        self._dispatch_locks = {name: threading.Lock() for name in self.experiments}
        self._audiences: dict[str, tuple[str, ...]] = {}
        self._order: deque[str] = deque()
        self._completed = {}
        self._descriptors = {name: {} for name in self.experiments}
        self._pending_subscriptions = {}
        self._epochs = {name: {} for name in self.experiments}
        self._committed_epoch = {name: 0 for name in self.experiments}
        self._submissions = {}
        self._start_epochs = {}
        self._source_started = False
        self._published = 0
        self._peak_bytes = 0
        self._peak_unpublished = 0
        self._error = None
        self._cancelled = {}
        self._abandoned_admissions = 0
        self._discarded_unpublished = 0
        self._eval_phase = False
        self._eval_names = ()
        self._eval_cursor = 0
        self._published_eval = 0

    def load_dataset(self, args) -> int:
        from aspec.controller.setup import auto_calculate_training_steps

        count = self.source.load_dataset(args)
        self.source.load_eval_dataset(args)
        for name, child in self.experiments.items():
            child._stored_dataset = self.source._stored_dataset
            child._dataset_fingerprint = self.source._dataset_fingerprint
            child._dataset_admission = dict(self.source._dataset_admission)
            raw_eval = self.source._stored_eval_dataset
            dp = int(getattr(child.args, "dp_size", 1))
            child._stored_eval_dataset = raw_eval[:len(raw_eval) // dp * dp]
            if raw_eval and not child._stored_eval_dataset:
                raise ValueError(f"Requested eval for {name} has fewer samples than dp_size={dp}")
            child._eval_dataset_admission = dict(self.source._eval_dataset_admission)
            if raw_eval:
                from aspec.controller.training_controller import dataset_fingerprint

                child._eval_dataset_admission.update(
                    admitted_samples=len(child._stored_eval_dataset),
                    truncated_samples=len(raw_eval) - len(child._stored_eval_dataset),
                    dataset_fingerprint=dataset_fingerprint(child._stored_eval_dataset),
                )
            auto_calculate_training_steps(child.args, count)
            self.remaining[name] = (None if getattr(child.args, "dflash_packing", False) and child.args.num_train_steps else child.args.num_train_steps * child.args.global_batch_size)
        return count

    def begin_eval(self, names):
        """Populate immutable held-out caches before independent training starts."""
        with self._lock:
            if self._source_started or self._eval_phase or self.get_pool_size() or self._order:
                raise RuntimeError("shared eval must precede the training source")
            if not names or not set(names).issubset(self.active):
                raise ValueError("shared eval requires active experiments")
            self._eval_phase, self._eval_names, self._eval_cursor = True, tuple(names), 0
            for name in names:
                child = self.experiments[name]
                child._eval_expected_count = len(child._stored_eval_dataset)
                child._eval_dispatched_samples = 0
            return max(len(self.experiments[n]._stored_eval_dataset) for n in names)

    def end_eval(self):
        with self._lock:
            if not self._eval_phase or self.get_pool_size() or self._order:
                raise RuntimeError("shared eval still owns source observations")
            for name in self._eval_names:
                self.experiments[name].finalize_eval_dispatch()
            self.source._eval_data_ids.clear()
            self._eval_phase, self._eval_names = False, ()

    def invoke(self, name, method, args, kwargs):
        if method in {"try_dispatch_batch", "try_dispatch_eval_batch"}:
            # Queue.put is a blocking Ray round trip. Independent experiments
            # must not hold the shared-stream lock while dispatching their own
            # queues; the retained controller protects its ledger/pool itself.
            with self._dispatch_locks[name]:
                with self._lock:
                    child = self.experiments[name]
                    if name not in self.active:
                        raise RuntimeError("cannot dispatch a retired experiment")
                    if name in self._cancelled:
                        raise RuntimeError(f"experiment cancelled: {self._cancelled[name]}")
                return getattr(child, method)(*args, **kwargs)
        with self._lock:
            child = self.experiments[name]
            if method == "check_running":
                if name in self._cancelled:
                    raise RuntimeError(f"experiment cancelled: {self._cancelled[name]}")
                if self._error:
                    raise RuntimeError(self._error)
                return None
            if method == "submit_training_dataset":
                if name not in self.active:
                    raise RuntimeError("cannot subscribe a retired experiment")
                submission = (args, kwargs)
                if name in self._submissions:
                    previous, result = self._submissions[name]
                    if submission != previous:
                        raise RuntimeError("cannot change an experiment's source subscription")
                    return result
                child_kwargs = dict(kwargs)
                completed_steps = child_kwargs.pop("completed_steps", 0)
                if (
                    not isinstance(completed_steps, int)
                    or not 0 <= completed_steps <= child.args.num_train_steps
                ):
                    raise ValueError("restored optimizer step exceeds the experiment horizon")
                if child_kwargs.get("committed_work_ids") and "completed_steps" not in kwargs:
                    raise ValueError("shared resume requires the restored optimizer step")
                result = child.submit_training_dataset(*args, **child_kwargs)
                # Old work may have used a different GBS. Remaining steps,
                # rather than cumulative sample count, define this attempt.
                steps = child.args.num_train_steps - completed_steps
                self.remaining[name] = None if getattr(child.args, "dflash_packing", False) and steps else steps * child.args.global_batch_size
                self.ready.add(name)
                self._start_epochs[name] = child._dataset_epoch
                self._committed_epoch[name] = child._dataset_epoch
                self._submissions[name] = (copy.deepcopy(submission), result)
                # Its prompt queue is not a second producer.
                child.prompt_buffer.clear()
                if not self._source_started and self.active.issubset(self.ready):
                    self.source.submit_training_dataset(
                        epoch=min(self._start_epochs[n] for n in self.active)
                    )
                    self._source_started = True
                return result
            if method == "reload_dataset":
                child._dataset_epoch += 1
                return len(child._stored_dataset)
            if method == "release_inference_output":
                result = child.release_inference_output(*args, **kwargs)
                record = child._lifecycle._records[kwargs["execution_id"]]
                if record.release_outcome is not None:
                    self._descriptors[name].pop(kwargs["execution_id"], None)
                return result
            if method == "commit_optimizer_step":
                result = child.commit_optimizer_step(*args, **kwargs)
                epochs = [
                    self._epochs[name].pop(identity, self._committed_epoch[name])
                    for identity in kwargs["execution_ids"]
                ]
                if epochs:
                    self._committed_epoch[name] = max(epochs)
                return result
            if method == "get_checkpoint_work_state":
                state = child.get_checkpoint_work_state()
                state["dataset_epoch"] = self._committed_epoch[name]
                return state
            return getattr(child, method)(*args, **kwargs)

    def get_prompts(self, count):
        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
            if self._eval_phase:
                limit = max(len(self.experiments[n]._stored_eval_dataset) for n in self._eval_names)
                stop = min(self._eval_cursor + count, limit)
                entries = self.source._build_eval_entries(self.source._stored_eval_dataset[self._eval_cursor:stop])
                for index, entry in enumerate(entries, self._eval_cursor):
                    identity = entry.metadata[EXECUTION_ID_KEY]
                    audience = tuple(n for n in self._eval_names if index < len(self.experiments[n]._stored_eval_dataset))
                    self._audiences[identity] = audience
                    self._order.append(identity)
                    for name in audience:
                        self.experiments[name]._eval_data_ids.add(entry.data_id)
                self._eval_cursor = stop
                return entries
            if not self.active.issubset(self.ready):
                return []
            entries = []
            while len(entries) < count and any(self.remaining[name] is None or self.remaining[name] > 0 for name in self.active):
                candidates = self.source.get_prompts(1)
                if not candidates:
                    self.source.reload_dataset()
                    continue
                entry = candidates[0]
                work_id = entry.metadata[WORK_ID_KEY]
                audience = tuple(
                    sorted(
                        name
                        for name in self.active
                        if (self.remaining[name] is None or self.remaining[name] > 0)
                        and self.source._dataset_epoch >= self._start_epochs[name]
                        and work_id not in self.experiments[name]._restored_committed_work_ids
                    )
                )
                if not audience:
                    continue
                entry.metadata["shared_source_epoch"] = self.source._dataset_epoch
                execution_id = entry.metadata[EXECUTION_ID_KEY]
                self._audiences[execution_id] = audience
                self._order.append(execution_id)
                for name in audience:
                    if self.remaining[name] is not None:
                        self.remaining[name] -= 1
                    self._epochs[name][execution_id] = self.source._dataset_epoch
                entries.append(entry)
            return entries

    def push_inference_results(self, results):
        with self._lock:
            for output in results:
                execution_id = output.metadata[EXECUTION_ID_KEY]
                if execution_id not in self._audiences or execution_id in self._completed:
                    raise RuntimeError("duplicate or unsolicited shared source output")
                self._completed[execution_id] = output
            self._peak_unpublished = max(self._peak_unpublished, len(self._completed))
            self._peak_bytes = max(self._peak_bytes, self.get_pool_bytes())
            # Target completion order must not change either experiment's data.
            while self._order and self._order[0] in self._completed:
                execution_id = self._order[0]
                output = self._completed[execution_id]
                descriptor = MooncakeP2PDescriptor.from_key(output.mooncake_key)
                audience = [n for n in self._audiences[execution_id] if n in self.active]
                if not audience:
                    source_control(descriptor)
                    self._order.popleft()
                    del self._completed[execution_id]
                    del self._audiences[execution_id]
                    continue
                views, readers, payloads = [], [], {}
                for name in audience:
                    args = self.experiments[name].args
                    cp_size = int(getattr(args, "draft_context_parallel_size", 1))
                    alignment = int(getattr(args, "draft_context_shard_alignment", 1))
                    source_contract = getattr(self.source.args, "aspec_feature_contract", None)
                    child_contract = getattr(args, "aspec_feature_contract", None)
                    view_options = {}
                    if source_contract is not None and child_contract is not None:
                        if source_contract["contract_id"] != child_contract["contract_id"]:
                            view_options = {
                                "source_layers": source_contract["aux_hidden_state_layers"],
                                "requested_layers": child_contract["aux_hidden_state_layers"],
                                "tensor_names": set(child_contract["tensors"]),
                            }
                    rank_views = shared_source_cp_views(
                        descriptor, name, cp_size, alignment=alignment, **view_options,
                    )
                    sample = copy.copy(output)
                    sample.mooncake_key = (rank_views[0].to_key() if cp_size == 1
                                           else pack_direct_cp_p2p_keys(v.to_key() for v in rank_views))
                    sample.metadata = dict(output.metadata, shared_source_experiment=name)
                    if view_options:
                        sample.tensor_shapes = {
                            tensor: tuple(output.tensor_shapes[tensor][:-1]) + (
                                child_contract["tensors"][tensor]["shape"][-1],
                            ) if tensor == "hidden_states" else output.tensor_shapes[tensor]
                            for tensor in child_contract["tensors"]
                        }
                        sample.tensor_dtypes = {
                            tensor: output.tensor_dtypes[tensor] for tensor in child_contract["tensors"]
                        }
                        if sample.metadata.get("aspec_feature_contract_id") != source_contract["contract_id"]:
                            raise RuntimeError("Shared source producer contract identity changed")
                        sample.metadata.update(
                            aspec_source_feature_contract_id=source_contract["contract_id"],
                            aspec_feature_contract_id=child_contract["contract_id"],
                            aspec_feature_view_layers=child_contract["aux_hidden_state_layers"],
                        )
                    if hasattr(args, "attention_backend"):
                        layout = "context_sharded" if cp_size > 1 else "full"
                        sample.metadata.update(
                            direct_cp_sharded=is_direct_cp_backend(
                                args.attention_backend
                            ),
                            direct_cp_size=cp_size, feature_layout=layout,
                            draft_attention_backend=args.attention_backend,
                            draft_context_parallel_size=cp_size,
                            draft_feature_layout=layout, context_shard_alignment=alignment,
                        )
                    readers.extend(v.reader_id for v in rank_views)
                    payloads.update({v.reader_id: v.payload_bytes for v in rank_views})
                    views.append((name, sample))
                # View construction is fallible. Before subscription, retain
                # only the unpublished physical owner, never partial readers.
                for name, _sample in views:
                    self._descriptors[name][execution_id] = descriptor
                self._reader_payloads[execution_id] = payloads
                # Track the frozen audience before the RPC: registration can
                # succeed remotely even when every response ACK is lost.
                self._pending_subscriptions[execution_id] = (descriptor, tuple(readers))
                self._confirm_subscription(execution_id)
                self._order.popleft()
                del self._completed[execution_id]
                del self._audiences[execution_id]
                # Every registered right is tracked before publishing to any
                # controller, so a partial fanout failure cannot orphan readers.
                for name, sample in views:
                    self.experiments[name].push_inference_results([sample])
                if self._eval_phase:
                    self._published_eval += 1
                else:
                    self._published += 1
            resident = self.get_pool_bytes()
            self._peak_bytes = max(self._peak_bytes, resident)
            return resident

    def _confirm_subscription(self, execution_id):
        pending = self._pending_subscriptions.get(execution_id)
        if pending is not None:
            descriptor, audience = pending
            payloads = self._reader_payloads[execution_id]
            options = ({"reader_payload_bytes": payloads}
                       if any(size != descriptor.payload_bytes for size in payloads.values()) else {})
            source_control(descriptor, operation="subscribe", readers=list(audience), **options)
            del self._pending_subscriptions[execution_id]
            self._reader_payloads.pop(execution_id, None)

    def get_ready_pool_size(self):
        with self._lock:
            # Completed outputs waiting for an earlier Target request are ready
            # storage too. Ignoring them admits more work through an apparently
            # empty pool until the physical ring fills, then bursts past its
            # intended watermark when the missing prefix completes.
            return max(
                (
                    self.experiments[n].get_ready_pool_size()
                    + sum(n in self._audiences[key] for key in self._completed)
                    for n in self.active
                ),
                default=0,
            )

    def get_pool_size(self):
        with self._lock:
            return len(
                set(self._completed)
                | {key for descriptors in self._descriptors.values() for key in descriptors}
            )

    def get_pool_bytes(self):
        with self._lock:
            unique = {
                key: value
                for descriptors in self._descriptors.values()
                for key, value in descriptors.items()
            }
            unique.update(
                {
                    key: MooncakeP2PDescriptor.from_key(output.mooncake_key)
                    for key, output in self._completed.items()
                }
            )
            return sum(descriptor.payload_bytes for descriptor in unique.values())

    def set_inference_error(self, message):
        with self._lock:
            self._error = str(message)
            for child in self.experiments.values():
                child.set_inference_error(message)

    def request_cancel(self, name, reason="requested"):
        """Stop new demand; the driver fences reads before withdrawing old rights."""
        with self._lock:
            if name not in self.experiments:
                raise ValueError(f"unknown experiment: {name}")
            if name in self.active:
                self._cancelled[name] = str(reason)
                self.remaining[name] = 0
            return {"experiment": name, "requested": name in self._cancelled}

    def cancellation_reason(self, name):
        with self._lock:
            return self._cancelled.get(name)

    def retire(self, name, *, owners_fenced: bool):
        """Called only after trainer reads ended or their owner processes exited."""

        if not owners_fenced:
            raise RuntimeError("cannot withdraw a reader while its owners may still read")
        with self._dispatch_locks[name], self._lock:
            if name in self._retired:
                return
            self.active.discard(name)
            self.remaining[name] = 0
            for execution_id, descriptor in self._descriptors[name].items():
                # Replaying the frozen subscription is idempotent whether the
                # first RPC was rejected or only its ACK was lost. Do not
                # withdraw an unknown reader or silently forget this output.
                self._confirm_subscription(execution_id)
                child_args = self.experiments[name].args
                cp_size = int(getattr(child_args, "draft_context_parallel_size", 1))
                for view in shared_source_cp_views(
                    descriptor, name, cp_size,
                    alignment=int(getattr(child_args, "draft_context_shard_alignment", 1)),
                ):
                    source_control(view, operation="withdraw", reader_id=view.reader_id)
            child = self.experiments[name]
            # Subscription can succeed before fanout fails. Reclaim every read
            # right, but only ledger work actually registered by this child.
            registered = [
                identity
                for identity in self._descriptors[name]
                if identity in child._lifecycle._records
            ]
            child._lifecycle.discard_after_owner_fence(registered)
            child.finalize_uncommitted_work()
            child.sample_pool.clear()
            child._sample_bytes.clear()
            child._pool_bytes = child._lifecycle.resident_bytes
            self._descriptors[name].clear()
            self._epochs[name].clear()
            # Queue actors belong to this experiment; no live trainer may use
            # them after the verified owner fence.
            for queue in getattr(child, "train_queues", []) + getattr(child, "eval_queues", []):
                queue.shutdown(force=True)
            self._retired.add(name)

    def close_queues(self):
        """Release session-owned queue actors after all reader groups are fenced."""
        with self._lock:
            if len(self._retired) != len(self.experiments):
                raise RuntimeError("cannot close queues while experiment owners are active")
            for queue in self.source.train_queues + self.source.eval_queues:
                queue.shutdown(force=True)

    def finish_production(self, *, inflight_samples):
        """Drop buffered admission only after producer drain and reader fencing."""
        with self._lock:
            if inflight_samples != 0 or len(self._retired) != len(self.experiments):
                raise RuntimeError(
                    "cannot finalize source while producer or read owners are active"
                )
            # Missing output proves no completed handoff, not that Target
            # necessarily did no computation before an RPC/backend failure.
            for execution_id, output in tuple(self._completed.items()):
                source_control(MooncakeP2PDescriptor.from_key(output.mooncake_key))
                del self._completed[execution_id]
                self._order.remove(execution_id)
                del self._audiences[execution_id]
                self._discarded_unpublished += 1
            self._abandoned_admissions += len(self._order)
            self._order.clear()
            self._completed.clear()
            self._audiences.clear()
            return self.status()

    def status(self):
        with self._lock:
            return {
                "published_source_observations": self._published,
                "published_eval_observations": self._published_eval,
                "active_experiments": sorted(self.active),
                "retiring_experiments": sorted(set(self.experiments) - self.active - self._retired),
                "outstanding_observations": self.get_pool_size(),
                "outstanding_payload_bytes": self.get_pool_bytes(),
                "peak_outstanding_payload_bytes": self._peak_bytes,
                "pending_production": len(self._order),
                "completed_unpublished_observations": len(self._completed),
                "pending_source_subscriptions": len(self._pending_subscriptions),
                "peak_completed_unpublished_observations": self._peak_unpublished,
                "remaining_source_demand": dict(self.remaining),
                "error": self._error,
                "cancelled_experiments": dict(self._cancelled),
                "abandoned_admissions_without_output": self._abandoned_admissions,
                "discarded_unpublished_observations": self._discarded_unpublished,
            }


SharedSourceController = ray.remote(max_concurrency=32)(SharedSourceState)
