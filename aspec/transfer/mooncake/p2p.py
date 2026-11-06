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

"""Opt-in Mooncake TransferEngine GPU P2P data plane.

This module deliberately does not implement CUDA IPC itself.  It owns registered
PyTorch CUDA allocations and asks Mooncake's ``nvlink_intra`` transport to move
the packed payload between them.  The descriptor is the only object that flows
through ASpec's existing controller/queue path.
"""

from __future__ import annotations

import base64
import ctypes
import importlib
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from aspec.utils.logging import logger

P2P_KEY_PREFIX = "aspec-p2p-v3:"
P2P_CP_KEY_PREFIX = "aspec-p2p-cp-v3:"
P2P_RELEASE_NOTIFY = "aspec-p2p-release-v3"
P2P_DESCRIPTOR_VERSION = 3
_RELEASE_MESSAGE_LIMIT = 16 * 1024
# Mooncake's TCP progress watchdog is per transfer_sync call.  A full long-
# context feature payload can be several GiB, so one call can exceed the
# watchdog when multiple DP ranks share a cross-node link even while bytes are
# moving.  Bound individual TCP operations without changing the descriptor or
# ownership unit.
_TCP_TRANSFER_CHUNK_BYTES = 128 * 1024 * 1024


def _import_mooncake_engine():
    """Import Mooncake's CUDA extension, resolving its wheel-local CUDA runtime.

    Mooncake CUDA wheels can be linked against CUDA 12 even when PyTorch uses a
    newer CUDA runtime.  The matching ``nvidia-cuda-runtime-cu12`` wheel keeps
    ``libcudart.so.12`` below site-packages, which is not necessarily present in
    ``LD_LIBRARY_PATH`` for Ray actors.  Load that exact process dependency by
    absolute path before retrying the extension import; do not alter the global
    library path or fall back to a different transport.
    """

    first_error: ImportError | None = None
    try:
        return importlib.import_module("mooncake.engine")
    except ImportError as exc:
        if "libcudart.so" not in str(exc):
            raise
        first_error = exc

    candidates: list[Path] = []
    for entry in sys.path:
        root = Path(entry or ".") / "nvidia" / "cuda_runtime" / "lib"
        exact = root / "libcudart.so.12"
        if exact.is_file():
            candidates.append(exact)
        elif root.is_dir():
            candidates.extend(sorted(root.glob("libcudart.so*")))

    load_errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return importlib.import_module("mooncake.engine")
        except (ImportError, OSError) as exc:
            load_errors.append(f"{candidate}: {exc}")

    detail = "; ".join(load_errors) if load_errors else "no wheel-local libcudart found"
    raise RuntimeError(
        "Mooncake TransferEngine import failed after resolving its CUDA runtime: "
        f"{detail}"
    ) from first_error


@dataclass(frozen=True)
class MooncakeP2PTensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    length: int
    # Shared CP views compact the destination but retain source byte ownership.
    source_offset: int | None = None
    layer_width: int | None = None  # logical [T, layers*D], physical [layers,T,D]
    source_regions: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MooncakeP2PTensorDescriptor":
        return cls(
            name=str(value["name"]),
            shape=tuple(int(dim) for dim in value["shape"]),
            dtype=str(value["dtype"]),
            offset=int(value["offset"]),
            length=int(value["length"]),
            source_offset=(int(value["source_offset"]) if value.get("source_offset") is not None else None),
            layer_width=(int(value["layer_width"]) if value.get("layer_width") is not None else None),
            source_regions=tuple(tuple(map(int, region)) for region in value.get("source_regions", ())),
        )


@dataclass(frozen=True)
class MooncakeP2PDescriptor:
    version: int
    run_id: str
    attempt_id: str
    store_key: str
    data_id: str
    request_id: str
    source_engine: str
    source_endpoint: str
    source_release_endpoint: str
    slot_base_ptr: int
    slot_id: int
    generation: int
    payload_bytes: int
    tensors: tuple[MooncakeP2PTensorDescriptor, ...]
    reader_id: str = ""
    backing_payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.version != P2P_DESCRIPTOR_VERSION:
            raise ValueError(f"unsupported Mooncake P2P descriptor version {self.version}")
        if not self.run_id or not self.attempt_id:
            raise ValueError("Mooncake P2P descriptor requires run_id and attempt_id")
        if not self.store_key or not self.source_endpoint:
            raise ValueError("Mooncake P2P descriptor requires store_key and source_endpoint")
        if self.slot_base_ptr <= 0 or self.slot_id < 0 or self.generation <= 0:
            raise ValueError("Mooncake P2P descriptor contains an invalid slot lease")
        if self.payload_bytes <= 0 or not self.tensors:
            raise ValueError("Mooncake P2P descriptor requires a non-empty payload")

        views = [tensor.source_offset is not None for tensor in self.tensors]
        if any(views) and not all(views):
            raise ValueError("P2P source offsets must describe every tensor or none")
        previous_end = 0
        names: set[str] = set()
        for tensor in sorted(self.tensors, key=lambda item: item.offset):
            if not tensor.name or tensor.name in names:
                raise ValueError(f"duplicate or empty P2P tensor name {tensor.name!r}")
            names.add(tensor.name)
            if (tensor.offset < previous_end or tensor.length < 0
                    or (tensor.length == 0 and tensor.source_offset is None)
                    or (tensor.source_offset is not None and tensor.source_offset < 0)):
                raise ValueError(f"invalid P2P tensor extent for {tensor.name}")
            if any(dim < 0 for dim in tensor.shape):
                raise ValueError(f"invalid P2P tensor shape for {tensor.name}")
            if tensor.layer_width is not None and (
                len(tensor.shape) != 2 or tensor.layer_width <= 0
                or tensor.shape[-1] % tensor.layer_width
            ):
                raise ValueError("invalid layer-major feature layout")
            if tensor.source_regions:
                if tensor.source_offset is None or sum(n for _, n in tensor.source_regions) != tensor.length:
                    raise ValueError("invalid P2P source view regions")
                if any(lo < 0 or n <= 0 for lo, n in tensor.source_regions):
                    raise ValueError("invalid P2P source region extent")
            if self.backing_payload_bytes is not None:
                regions = tensor.source_regions or ((tensor.source_offset, tensor.length),)
                if self.backing_payload_bytes <= 0 or any(
                    lo is None or lo < 0 or lo + n > self.backing_payload_bytes for lo, n in regions
                ):
                    raise ValueError("P2P source view exceeds backing allocation")
            previous_end = tensor.offset + tensor.length
        if previous_end > self.payload_bytes:
            raise ValueError("P2P tensor extents exceed payload_bytes")

    def to_key(self) -> str:
        payload = asdict(self)
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        return P2P_KEY_PREFIX + encoded.rstrip("=")

    @classmethod
    def from_key(cls, key: str) -> "MooncakeP2PDescriptor":
        if not isinstance(key, str) or not key.startswith(P2P_KEY_PREFIX):
            raise ValueError("not a ASpec Mooncake P2P descriptor key")
        encoded = key[len(P2P_KEY_PREFIX) :]
        try:
            encoded += "=" * (-len(encoded) % 4)
            value = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
            value["tensors"] = tuple(
                MooncakeP2PTensorDescriptor.from_dict(item) for item in value["tensors"]
            )
            return cls(**value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ASpec Mooncake P2P descriptor key") from exc


@dataclass(frozen=True)
class RingLease:
    slot_id: int
    generation: int
    owner: str


@dataclass(frozen=True)
class MooncakeP2PReadResult:
    tensors: dict[str, torch.Tensor]
    lease: "MooncakeP2PReceiveLease"


class MooncakeP2PReceiveLease:
    """Own a destination ring slot until its trainer microstep is complete."""

    def __init__(
        self,
        transfer: "MooncakeP2PTransfer",
        lease: RingLease,
        descriptor: MooncakeP2PDescriptor,
    ) -> None:
        self._transfer = transfer
        self._lease = lease
        self._descriptor = descriptor
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError(
                    "stale or duplicate Mooncake P2P destination lease release: "
                    f"slot={self._lease.slot_id}, generation={self._lease.generation}"
                )
            self._released = True
        self._transfer._release_destination_lease(self._lease, self._descriptor)


class MooncakeP2PReservation:
    """A source-ring slot whose tensor views can be filled by model execution."""

    def __init__(
        self,
        transfer: "MooncakeP2PTransfer",
        lease: RingLease,
        descriptor: MooncakeP2PDescriptor,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self._transfer = transfer
        self._lease = lease
        self.descriptor = descriptor
        self.tensors = tensors
        self._lock = threading.Lock()
        self._state = "reserved"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def commit(self) -> MooncakeP2PDescriptor:
        """Wait for producer-stream writes before exposing the descriptor."""

        with self._lock:
            if self._state != "reserved":
                raise RuntimeError(
                    f"cannot commit Mooncake P2P reservation in state {self._state!r}"
                )
            self._transfer._commit_reservation(self)
            self._state = "committed"
        return self.descriptor

    def discard(self) -> None:
        """Release an unpublished reservation, including one already committed."""

        with self._lock:
            if self._state == "discarded":
                return
            if self._state == "published":
                raise RuntimeError("cannot discard a published Mooncake P2P reservation")
            self._transfer._discard_reservation(self)
            self._state = "discarded"

    def publish(self) -> None:
        """Transfer ownership to the consumer after its descriptor is emitted."""

        with self._lock:
            if self._state != "committed":
                raise RuntimeError(
                    f"cannot publish Mooncake P2P reservation in state {self._state!r}"
                )
            self._state = "published"


class GenerationSlotRing:
    """Thread-safe slot state machine with strict generation/owner checks."""

    def __init__(self, slots: int):
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self._condition = threading.Condition()
        self._generations = [0] * slots
        self._leases: list[RingLease | None] = [None] * slots
        self._acknowledged_releases: deque[RingLease] = deque()
        self._acknowledged_release_set: set[RingLease] = set()
        self._acknowledged_release_limit = max(64, slots * 32)
        self._readers: dict[RingLease, tuple[frozenset[str], set[str]]] = {}

    @property
    def slots(self) -> int:
        return len(self._leases)

    @property
    def in_use(self) -> int:
        with self._condition:
            return sum(lease is not None for lease in self._leases)

    def wait_until_empty(self, timeout: float | None = None) -> bool:
        """Wait for every outstanding lease to return to the ring."""

        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while any(lease is not None for lease in self._leases):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def acquire(self, owner: str, timeout: float | None = None) -> RingLease:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while True:
                for slot_id, current in enumerate(self._leases):
                    if current is None:
                        self._generations[slot_id] += 1
                        lease = RingLease(slot_id, self._generations[slot_id], owner)
                        self._leases[slot_id] = lease
                        return lease
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for a GPU ring slot ({self.slots} slots busy)"
                    )
                self._condition.wait(remaining)

    def release(self, slot_id: int, generation: int, owner: str) -> None:
        with self._condition:
            if slot_id < 0 or slot_id >= len(self._leases):
                raise RuntimeError(f"stale P2P release for invalid slot {slot_id}")
            expected = self._leases[slot_id]
            actual = RingLease(slot_id, generation, owner)
            if expected != actual:
                raise RuntimeError(
                    "stale or duplicate P2P slot release: "
                    f"expected={expected!r}, received={actual!r}"
                )
            if actual in self._readers and self._readers[actual][1]:
                raise RuntimeError("cannot release a source slot with pending readers")
            self._leases[slot_id] = None
            self._condition.notify()

    def register_readers(self, lease: RingLease, readers: list[str]) -> bool:
        """Freeze the readers before any shared descriptor becomes visible."""

        audience = frozenset(readers)
        if not audience or len(audience) != len(readers) or any(not r for r in audience):
            raise ValueError("source readers must be unique non-empty identities")
        with self._condition:
            if lease.slot_id < 0 or lease.slot_id >= self.slots:
                raise RuntimeError("invalid shared source slot")
            if self._leases[lease.slot_id] != lease:
                raise RuntimeError("cannot subscribe to a stale source generation")
            previous = self._readers.get(lease)
            if previous is not None:
                if previous[0] != audience:
                    raise RuntimeError("source reader set cannot change after publication")
                return False
            self._readers[lease] = (audience, set(audience))
            return True

    def reader_pending(self, lease: RingLease, reader: str) -> bool:
        with self._condition:
            registered = self._readers.get(lease)
            if registered is None or reader not in registered[0]:
                raise RuntimeError("unknown or stale shared source reader")
            return reader in registered[1]

    def release_reader(self, lease: RingLease, reader: str) -> bool:
        """Acknowledge one reader; only the last reader returns the source slot."""

        with self._condition:
            registered = self._readers.get(lease)
            if registered is None or reader not in registered[0]:
                raise RuntimeError("unknown or stale shared source reader")
            pending = registered[1]
            if reader not in pending:
                return False
            if self._leases[lease.slot_id] != lease:
                raise RuntimeError("shared source generation changed before read completed")
            pending.remove(reader)
            if pending:
                return False
            return self.release_acknowledged(lease.slot_id, lease.generation, lease.owner)

    def release_acknowledged(
        self, slot_id: int, generation: int, owner: str
    ) -> bool:
        """Release once and accept a bounded retry of the same acknowledged lease.

        The source mutates ownership before returning the control-plane ACK.  If
        that ACK is lost, a destination retry must not fail the source or release
        a newer generation that already reused the slot.
        """

        with self._condition:
            if slot_id < 0 or slot_id >= len(self._leases):
                raise RuntimeError(f"stale P2P release for invalid slot {slot_id}")
            actual = RingLease(slot_id, generation, owner)
            expected = self._leases[slot_id]
            if expected == actual:
                if actual in self._readers and self._readers[actual][1]:
                    raise RuntimeError("cannot release a source slot with pending readers")
                self._leases[slot_id] = None
                if len(self._acknowledged_releases) >= self._acknowledged_release_limit:
                    expired = self._acknowledged_releases.popleft()
                    self._acknowledged_release_set.discard(expired)
                    self._readers.pop(expired, None)
                self._acknowledged_releases.append(actual)
                self._acknowledged_release_set.add(actual)
                self._condition.notify()
                return True
            if actual in self._acknowledged_release_set:
                return False
            raise RuntimeError(
                "stale or foreign P2P acknowledged release: "
                f"expected={expected!r}, received={actual!r}"
            )


def is_p2p_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(P2P_KEY_PREFIX)


def p2p_descriptors_from_key(key: str) -> tuple[MooncakeP2PDescriptor, ...]:
    """Return every source-ring descriptor represented by a transport key."""

    if is_p2p_key(key):
        return (MooncakeP2PDescriptor.from_key(key),)
    if not isinstance(key, str) or not key.startswith(P2P_CP_KEY_PREFIX):
        raise ValueError("not a ASpec Mooncake P2P descriptor or CP bundle key")

    encoded = key[len(P2P_CP_KEY_PREFIX) :]
    try:
        encoded += "=" * (-len(encoded) % 4)
        values = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid ASpec Direct CP P2P bundle key") from exc
    if not isinstance(values, list) or not values:
        raise ValueError(
            "ASpec Direct CP P2P bundle requires non-empty descriptors, got "
            f"{len(values) if isinstance(values, list) else 'invalid'}"
        )
    descriptors = tuple(MooncakeP2PDescriptor.from_key(value) for value in values)
    for rank, descriptor in enumerate(descriptors):
        if not descriptor.store_key.endswith(f"_cp{rank}") and not (
            descriptor.reader_id.endswith(f"/cp/{rank}")
            and all(t.source_offset is not None for t in descriptor.tensors)
        ):
            raise ValueError(
                "Direct CP P2P descriptor order does not match its shard rank: "
                f"rank={rank}, store_key={descriptor.store_key!r}"
            )
    return descriptors


def pack_direct_cp_p2p_keys(keys: Iterable[str]) -> str:
    """Serialize the per-rank P2P descriptors for one Direct CP sample."""

    values = tuple(keys)
    if not values:
        raise ValueError(f"Direct CP P2P bundle requires non-empty keys, got {len(values)}")
    descriptors = tuple(MooncakeP2PDescriptor.from_key(key) for key in values)
    for rank, descriptor in enumerate(descriptors):
        if not descriptor.store_key.endswith(f"_cp{rank}") and not (
            descriptor.reader_id.endswith(f"/cp/{rank}")
            and all(t.source_offset is not None for t in descriptor.tensors)
        ):
            raise ValueError(
                "Direct CP P2P descriptor order does not match its shard rank: "
                f"rank={rank}, store_key={descriptor.store_key!r}"
            )
    encoded = base64.urlsafe_b64encode(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return P2P_CP_KEY_PREFIX + encoded.rstrip("=")


def direct_cp_p2p_key_for_rank(key: str, *, cp_rank: int, cp_size: int) -> str:
    """Resolve a Direct CP sample key to the descriptor owned by ``cp_rank``."""

    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid Direct CP rank {cp_rank}/{cp_size}")
    # A CP1 direct-output reservation already contains the complete sample and
    # therefore needs no synthetic ``_cp0`` object key.
    if cp_size == 1 and is_p2p_key(key):
        return key
    if not isinstance(key, str) or not key.startswith(P2P_CP_KEY_PREFIX):
        return f"{key}_cp{cp_rank}"
    encoded = key[len(P2P_CP_KEY_PREFIX) :]
    try:
        encoded += "=" * (-len(encoded) % 4)
        values = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid ASpec Direct CP P2P bundle key") from exc
    if not isinstance(values, list) or len(values) != cp_size:
        raise ValueError(
            "Direct CP P2P bundle size does not match the process group: "
            f"bundle={len(values) if isinstance(values, list) else 'invalid'}, cp_size={cp_size}"
        )
    descriptor_key = values[cp_rank]
    descriptor = MooncakeP2PDescriptor.from_key(descriptor_key)
    if not descriptor.store_key.endswith(f"_cp{cp_rank}") and not (
        descriptor.reader_id.endswith(f"/cp/{cp_rank}")
        and all(t.source_offset is not None for t in descriptor.tensors)
    ):
        raise ValueError(
            "Direct CP P2P descriptor does not match the receiving rank: "
            f"rank={cp_rank}, store_key={descriptor.store_key!r}"
        )
    return descriptor_key


def shared_source_cp_views(descriptor, experiment, cp_size, *, alignment=1,
                           source_layers=None, requested_layers=None, tensor_names=None):
    """Compact selected layers and CP rows without copying or restaging source."""
    from aspec.inference.context import context_shard_bounds

    if cp_size <= 0:
        raise ValueError("CP view size must be positive")
    select_layers = requested_layers is not None
    if cp_size == 1 and not select_layers and tensor_names is None:
        return (replace(descriptor, reader_id=experiment),)
    columns = None
    if select_layers:
        if source_layers is None or len(set(requested_layers)) != len(requested_layers):
            raise ValueError("Feature view requires unique source/requested layers")
        try:
            columns = [source_layers.index(layer) for layer in requested_layers]
        except ValueError as exc:
            raise ValueError("Requested feature layer is absent from source") from exc
    ids = next(t for t in descriptor.tensors if t.name == "input_ids")
    length = ids.shape[-1]
    views = []
    for rank in range(cp_size):
        start, end = context_shard_bounds(length, rank, cp_size, alignment=alignment)
        tensors, cursor = [], 0
        for tensor in descriptor.tensors:
            if tensor_names is not None and tensor.name not in tensor_names:
                continue
            cursor = _align(cursor)
            shape, source_offset, size = list(tensor.shape), tensor.offset, tensor.length
            regions = ()
            if tensor.name != "input_ids":
                if len(shape) not in (2, 3) or (len(shape) == 3 and shape[0] != 1):
                    raise ValueError("CP feature view requires token-major single-document tensors")
                if shape[-2] != length or tensor.length % length:
                    raise ValueError("CP feature extent does not match replicated token IDs")
                shape[-2] = end - start
                if tensor.layer_width is not None:
                    width = tensor.layer_width
                    count = tensor.shape[-1] // width
                    selected = columns if tensor.name == "hidden_states" and columns is not None else list(range(count))
                    if count != len(source_layers or range(count)) or any(c >= count for c in selected):
                        raise ValueError("Layer-major source does not match its layer contract")
                    element_size = torch.empty((), dtype=_torch_dtype(tensor.dtype)).element_size()
                    row_bytes = width * element_size
                    regions = tuple((tensor.offset + (c * length + start) * row_bytes,
                                     (end - start) * row_bytes) for c in selected if end > start)
                    size = len(selected) * (end - start) * row_bytes
                    shape[-1] = len(selected) * width
                else:
                    if tensor.name == "hidden_states" and columns is not None and (
                        columns != list(range(len(source_layers))) or requested_layers != source_layers
                    ):
                        raise ValueError("Layer selection requires layer-major registered source storage")
                    row_bytes = tensor.length // length
                    source_offset += start * row_bytes
                    size = (end - start) * row_bytes
            tensors.append(replace(tensor, shape=tuple(shape), offset=cursor,
                                   length=size, source_offset=source_offset, source_regions=regions))
            cursor += size
        views.append(replace(
            descriptor, payload_bytes=max(1, cursor), tensors=tuple(tensors),
            reader_id=experiment if cp_size == 1 else f"{experiment}/cp/{rank}",
            backing_payload_bytes=descriptor.payload_bytes,
        ))
    return tuple(views)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.replace("torch.", ""), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype in P2P descriptor: {name!r}")
    return dtype


def _align(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _build_tensor_layout(
    tensors: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
    *,
    capacity: int,
) -> tuple[tuple[MooncakeP2PTensorDescriptor, ...], int]:
    descriptors: list[MooncakeP2PTensorDescriptor] = []
    names: set[str] = set()
    offset = 0
    for name, shape, dtype in tensors:
        if not name or name in names:
            raise ValueError(f"duplicate or empty P2P tensor name {name!r}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"P2P tensor {name} has invalid dtype {dtype!r}")
        shape = tuple(int(dim) for dim in shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"P2P tensor {name} has invalid shape {shape!r}")
        names.add(name)
        numel = 1
        for dim in shape:
            numel *= dim
        offset = _align(offset)
        length = numel * torch.empty((), dtype=dtype).element_size()
        end = offset + length
        if end > capacity:
            raise RuntimeError(
                f"P2P payload exceeds slot capacity: need={end}, capacity={capacity}"
            )
        descriptors.append(
            MooncakeP2PTensorDescriptor(
                name=name,
                shape=shape,
                dtype=_dtype_name(dtype),
                offset=offset,
                length=length,
            )
        )
        offset = end
    if not descriptors:
        raise ValueError("P2P reservation requires at least one tensor")
    return tuple(descriptors), offset


class MooncakeP2PTransfer:
    """Registered CUDA slot rings backed by Mooncake TransferEngine."""

    def __init__(self, config, *, timeline=None):
        self.config = config
        self.timeline = timeline
        self.engine = None
        self.endpoint = ""
        self.release_endpoint = ""
        self.device: torch.device | None = None
        self.role = ""
        self.ring: GenerationSlotRing | None = None
        self.buffers: list[torch.Tensor] = []
        self._ring_buffer: torch.Tensor | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self._stop = threading.Event()
        self._notify_thread: threading.Thread | None = None
        self._release_server: socket.socket | None = None
        self._release_thread: threading.Thread | None = None
        self._last_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._shared_payload_bytes: dict[RingLease, int] = {}
        self._shared_completed_readers: dict[str, dict[str, int]] = {}
        self._shared_reader_bytes = {}
        self._started_monotonic: float | None = None
        self._metrics: dict[str, int | float] = {
            "ring_wait_seconds": 0.0,
            "source_descriptors": 0,
            "source_payload_bytes": 0,
            "source_releases": 0,
            "discarded_source_payload_bytes": 0,
            "destination_reads": 0,
            "destination_transfer_chunks": 0,
            "destination_payload_bytes": 0,
            "destination_transfer_seconds": 0.0,
            "destination_max_transfer_seconds": 0.0,
            "destination_releases": 0,
            "transfer_failures": 0,
            "discarded_source_descriptors": 0,
            "source_release_requests": 0,
            "source_release_duplicates": 0,
            "shared_expected_reads": 0,
            "shared_expected_payload_bytes": 0,
            "shared_withdrawn_reads": 0,
            "shared_withdrawn_payload_bytes": 0,
            "destination_release_acks": 0,
            "destination_release_retries": 0,
            "destination_release_failures": 0,
        }

    def _read_payload(
        self,
        descriptor: MooncakeP2PDescriptor,
        *,
        destination_ptr: int,
        opcode: Any,
        notify: Any,
    ) -> tuple[int, int]:
        """Read one descriptor, bounding TCP calls below its progress watchdog."""

        regions = []
        if any(t.source_offset is not None for t in descriptor.tensors):
            for tensor in descriptor.tensors:
                if not tensor.length:
                    continue
                offset = tensor.offset
                for source, length in tensor.source_regions or ((tensor.source_offset, tensor.length),):
                    regions.append((offset, source, length))
                    offset += length
        else:
            regions = [(0, 0, descriptor.payload_bytes)]
        chunks = 0
        for index, (destination_offset, source_offset, size) in enumerate(regions):
            offset = 0
            while offset < size:
                length = min(_TCP_TRANSFER_CHUNK_BYTES, size - offset) if self.config.protocol == "tcp" else size - offset
                final = index == len(regions) - 1 and offset + length == size
                rc = self.engine.transfer_sync(
                    descriptor.source_endpoint, destination_ptr + destination_offset + offset,
                    descriptor.slot_base_ptr + source_offset + offset, length, opcode,
                    notify if final else None,
                )
                chunks += 1
                if rc != 0:
                    return rc, chunks
                offset += length
        return 0, chunks

    def setup(self, device: torch.device, *, role: str) -> None:
        if role not in ("source", "destination"):
            raise ValueError("transfer_engine setup role must be 'source' or 'destination'")
        if not torch.cuda.is_available():
            raise RuntimeError("transfer_engine data_plane requires CUDA")
        if not self.config.gpu_buffer_size:
            raise ValueError("transfer_engine data_plane requires gpu_buffer_size")

        self.config.apply_env_defaults()
        self.device = torch.device(device)
        self.role = role
        torch.cuda.set_device(self.device)

        mooncake_engine = _import_mooncake_engine()

        if not getattr(mooncake_engine, "SUPPORT_CUDA", False):
            raise RuntimeError("Mooncake wheel was built without CUDA support")
        if self.config.require_nvlink and not getattr(
            mooncake_engine, "SUPPORT_INTRA_NVLINK", False
        ):
            raise RuntimeError("Mooncake wheel was built without intra-node NVLink support")
        self.engine = mooncake_engine.TransferEngine()
        hostname = self.config.local_hostname or "localhost"
        device_name = self.config.device_name if self.config.protocol == "rdma" else ""
        rc = self.engine.initialize(
            hostname, "P2PHANDSHAKE", self.config.protocol, device_name
        )
        if rc != 0:
            raise RuntimeError(
                "Mooncake TransferEngine initialization failed "
                f"(protocol={self.config.protocol}, rc={rc})"
            )
        self.endpoint = f"{hostname}:{self.engine.get_rpc_port()}"
        self._started_monotonic = time.monotonic()

        ring_slots = self.config.ring_slots_for_role(role)
        self.ring = GenerationSlotRing(ring_slots)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # NVLink registers the underlying cudaMalloc allocation, not each
        # PyTorch suballocation. Own one backing tensor and register/unregister
        # it once; slot views retain independent generation/lease identities.
        slot_bytes = int(self.config.gpu_buffer_size)
        buffer = torch.empty(
            ring_slots * slot_bytes, dtype=torch.uint8, device=self.device
        )
        rc = self.engine.register_memory(
            buffer.data_ptr(), buffer.numel(), f"cuda:{self.device.index or 0}"
        )
        if rc != 0:
            self.close()
            raise RuntimeError(f"Mooncake GPU ring memory registration failed (rc={rc})")
        self._ring_buffer = buffer
        self.buffers = [
            buffer.narrow(0, slot_id * slot_bytes, slot_bytes)
            for slot_id in range(ring_slots)
        ]

        if role == "source":
            if self.config.protocol in {"tcp", "rdma"} or os.getenv("ASPEC_SHARED_SOURCE") == "1":
                try:
                    self._start_release_server(hostname)
                except BaseException:
                    self.close()
                    raise
            else:
                self._notify_thread = threading.Thread(
                    target=self._poll_notifies,
                    name="mooncake-p2p-notify",
                    daemon=True,
                )
                self._notify_thread.start()

        logger.info(
            "Mooncake TransferEngine P2P initialized: endpoint=%s protocol=%s role=%s "
            "release_endpoint=%s slots=%d slot_bytes=%d SUPPORT_INTRA_NVLINK=%s",
            self.endpoint,
            self.config.protocol,
            role,
            self.release_endpoint or "native_notify",
            ring_slots,
            self.config.gpu_buffer_size,
            getattr(mooncake_engine, "SUPPORT_INTRA_NVLINK", False),
        )

    def check_last_error(self) -> None:
        with self._error_lock:
            error = self._last_error
        if error is not None:
            raise RuntimeError("Mooncake P2P background worker failed") from error

    def _start_release_server(self, hostname: str) -> None:
        """Start the acknowledged source-lease control plane.

        Mooncake's ``transfer_sync`` return code covers the data operation but
        not delivery of its optional notification.  Cross-node ownership must
        therefore use a control operation whose success is visible to both
        processes.
        """

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((hostname, 0))
        server.listen(max(16, (self.ring.slots if self.ring is not None else 1) * 2))
        server.settimeout(0.2)
        self._release_server = server
        self.release_endpoint = f"{hostname}:{server.getsockname()[1]}"
        self._release_thread = threading.Thread(
            target=self._serve_release_requests,
            name="aspec-p2p-release",
            daemon=True,
        )
        self._release_thread.start()

    @staticmethod
    def _recv_release_message(connection: socket.socket) -> dict[str, Any]:
        payload = bytearray()
        while b"\n" not in payload:
            chunk = connection.recv(min(4096, _RELEASE_MESSAGE_LIMIT - len(payload)))
            if not chunk:
                raise ConnectionError("release peer closed before sending a request")
            payload.extend(chunk)
            if len(payload) >= _RELEASE_MESSAGE_LIMIT and b"\n" not in payload:
                raise ValueError("P2P release request exceeds the message limit")
        line, _separator, remainder = payload.partition(b"\n")
        if remainder:
            raise ValueError("P2P release connection contains trailing data")
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("P2P release request must be a JSON object")
        return value

    def _apply_release_request(self, value: dict[str, Any]) -> bool:
        if self.ring is None:
            raise RuntimeError("Mooncake P2P source ring is closed")
        expected_run_id = os.getenv("ASPEC_RUN_ID", "")
        expected_attempt_id = os.getenv("ASPEC_ATTEMPT_ID", "")
        if expected_run_id and value.get("run_id") != expected_run_id:
            raise RuntimeError("Mooncake P2P release belongs to a stale or foreign run")
        if expected_attempt_id and value.get("attempt_id") != expected_attempt_id:
            raise RuntimeError(
                "Mooncake P2P release belongs to a stale or foreign attempt"
            )
        lease = RingLease(int(value["slot_id"]), int(value["generation"]), str(value["store_key"]))
        if value.get("operation") == "subscribe":
            reader_bytes = value.get("reader_payload_bytes") or {
                reader: int(value["payload_bytes"]) for reader in value["readers"]
            }
            if set(reader_bytes) != set(value["readers"]) or any(
                not isinstance(n, int) or isinstance(n, bool) or n <= 0
                for n in reader_bytes.values()
            ):
                raise ValueError("Invalid shared reader payload sizes")
            previous = self._shared_reader_bytes.get(lease)
            if previous is not None and previous != reader_bytes:
                raise RuntimeError("Shared reader payload sizes cannot change after publication")
            first = self.ring.register_readers(lease, value["readers"])
            if first:
                self._shared_reader_bytes[lease] = dict(reader_bytes)
                self._add_metric("shared_expected_reads", len(value["readers"]))
                self._add_metric("shared_expected_payload_bytes", sum(reader_bytes.values()))
                self._shared_payload_bytes[lease] = int(value["payload_bytes"])
            return False
        if value.get("reader_id"):
            first = self.ring.reader_pending(lease, str(value["reader_id"]))
            released = self.ring.release_reader(lease, str(value["reader_id"]))
            reader_bytes = self._shared_reader_bytes.get(lease, {}).get(
                str(value["reader_id"]), self._shared_payload_bytes.get(lease, int(value.get("payload_bytes", 0)))
            )
            if value.get("operation") == "withdraw" and first:
                self._add_metric("shared_withdrawn_reads")
                self._add_metric("shared_withdrawn_payload_bytes", reader_bytes)
            elif first:
                with self._metrics_lock:
                    completed = self._shared_completed_readers.setdefault(str(value["reader_id"]),
                                                                         {"reads": 0, "payload_bytes": 0})
                    completed["reads"] += 1
                    completed["payload_bytes"] += reader_bytes
            if released:
                self._shared_payload_bytes.pop(lease, None)
                self._shared_reader_bytes.pop(lease, None)
        else:
            released = self.ring.release_acknowledged(lease.slot_id, lease.generation, lease.owner)
        if released:
            self._add_metric("source_releases")
            self._add_metric("source_release_requests")
            if self.timeline is not None:
                self.timeline.event(
                    "gpu_slot_release",
                    mooncake_key=str(value["store_key"]),
                    slot_id=int(value["slot_id"]),
                    generation=int(value["generation"]),
                    ring_role="source_acknowledged",
                    data_id=value.get("data_id"),
                    request_id=value.get("request_id"),
                    source_engine=value.get("source_engine"),
                )
        else:
            self._add_metric("source_release_duplicates")
        return released

    def _serve_release_requests(self) -> None:
        server = self._release_server
        if server is None:
            return
        try:
            while not self._stop.is_set():
                try:
                    connection, _peer = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stop.is_set():
                        return
                    raise
                with connection:
                    connection.settimeout(5.0)
                    try:
                        value = self._recv_release_message(connection)
                        released = self._apply_release_request(value)
                        response = {"ok": True, "released": released}
                    except (ConnectionError, TimeoutError, json.JSONDecodeError) as exc:
                        logger.warning("Ignored incomplete P2P release request: %s", exc)
                        continue
                    except BaseException as exc:
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        try:
                            connection.sendall(
                                json.dumps(response, separators=(",", ":")).encode(
                                    "utf-8"
                                )
                                + b"\n"
                            )
                        except OSError:
                            pass
                        raise
                    try:
                        connection.sendall(
                            json.dumps(response, separators=(",", ":")).encode("utf-8")
                            + b"\n"
                        )
                    except OSError:
                        # Ownership already changed.  A client whose ACK was
                        # lost retries the same generation and receives an
                        # idempotent success response.
                        logger.warning("P2P source release ACK could not be delivered")
        except BaseException as exc:
            with self._error_lock:
                self._last_error = exc
            logger.exception("Mooncake P2P release server failed")

    def _acknowledge_source_release(self, descriptor: MooncakeP2PDescriptor) -> None:
        endpoint = descriptor.source_release_endpoint
        if not endpoint:
            raise RuntimeError("cross-node P2P descriptor has no release endpoint")
        try:
            host, raw_port = endpoint.rsplit(":", maxsplit=1)
            port = int(raw_port)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"invalid P2P source release endpoint {endpoint!r}") from exc

        message = json.dumps(
            {
                "store_key": descriptor.store_key,
                "run_id": descriptor.run_id,
                "attempt_id": descriptor.attempt_id,
                "slot_id": descriptor.slot_id,
                "generation": descriptor.generation,
                "data_id": descriptor.data_id,
                "request_id": descriptor.request_id,
                "source_engine": descriptor.source_engine,
                "reader_id": descriptor.reader_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        max_wait = max(float(self.config.get_retry_max_wait_seconds), 0.0)
        deadline = time.monotonic() + max_wait
        attempt = 0
        last_error: BaseException | None = None
        while True:
            attempt += 1
            remaining = max(0.0, deadline - time.monotonic())
            timeout = min(5.0, remaining) if max_wait > 0 else 5.0
            try:
                with socket.create_connection((host, port), timeout=timeout) as connection:
                    connection.settimeout(timeout)
                    connection.sendall(message)
                    response = self._recv_release_message(connection)
                if response.get("ok") is not True:
                    raise RuntimeError(
                        "P2P source rejected lease release: "
                        f"{response.get('error', 'unknown error')}"
                    )
                self._add_metric("destination_release_acks")
                if attempt > 1:
                    self._add_metric("destination_release_retries", attempt - 1)
                return
            except RuntimeError:
                self._add_metric("destination_release_failures")
                self._add_metric("transfer_failures")
                raise
            except (ConnectionError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if max_wait <= 0 or time.monotonic() >= deadline:
                    self._add_metric("destination_release_failures")
                    self._add_metric("transfer_failures")
                    if attempt > 1:
                        self._add_metric("destination_release_retries", attempt - 1)
                    raise RuntimeError(
                        "P2P source lease release was not acknowledged "
                        f"(endpoint={endpoint}, attempts={attempt})"
                    ) from last_error
                time.sleep(min(0.05 * attempt, 0.5))

    def _add_metric(self, name: str, value: int | float = 1) -> None:
        with self._metrics_lock:
            self._metrics[name] += value

    def status(self) -> dict[str, Any]:
        """Return bounded transport facts for distributed Evidence."""

        with self._metrics_lock:
            metrics = dict(self._metrics)
            if self._shared_completed_readers:
                metrics["shared_completed_readers"] = {
                    name: dict(value) for name, value in self._shared_completed_readers.items()}
        with self._error_lock:
            notification_error = self._last_error
        transfer_seconds = float(metrics["destination_transfer_seconds"])
        payload_bytes = int(metrics["destination_payload_bytes"])
        metrics["destination_payload_bytes_per_second"] = (
            payload_bytes / transfer_seconds if transfer_seconds > 0 else None
        )
        elapsed = (
            time.monotonic() - self._started_monotonic
            if self._started_monotonic is not None
            else 0.0
        )
        return {
            "state": "active" if self.engine is not None else "closed",
            "role": self.role,
            "protocol": self.config.protocol,
            "endpoint": self.endpoint,
            "release_endpoint": self.release_endpoint or None,
            "strict_transport": not self.config.allow_host_fallback,
            "activation": (
                "strict_rdma_initialized"
                if self.config.protocol == "rdma"
                and not self.config.allow_host_fallback
                else "initialized"
            ),
            "node_address": getattr(self.config, "local_hostname", ""),
            "run_id": os.getenv("ASPEC_RUN_ID", ""),
            "attempt_id": os.getenv("ASPEC_ATTEMPT_ID", ""),
            "configured_rdma_route": (
                {
                    "device_name": self.config.device_name,
                    "port": self.config.rdma_port,
                    "gid_index": self.config.rdma_gid_index,
                }
                if self.config.protocol == "rdma"
                else None
            ),
            # Mooncake currently exposes initialization success but not the
            # native route selected by TransferEngine.  Do not relabel the
            # canonical configuration as a machine-observed fact.
            "observed_rdma_route": None,
            "notification_worker_error": (
                f"{type(notification_error).__name__}: {notification_error}"
                if notification_error is not None
                else None
            ),
            "elapsed_seconds": elapsed,
            "ring": {
                "slots": self.ring.slots if self.ring is not None else 0,
                "in_use": self.ring.in_use if self.ring is not None else 0,
                "slot_bytes": int(self.config.gpu_buffer_size or 0),
            },
            "metrics": metrics,
        }

    def wait_for_source_releases(self, timeout: float = 5.0) -> bool:
        """Keep the source endpoint alive until consumers return its slots."""

        if self.role != "source" or self.ring is None:
            return True
        self.check_last_error()
        drained = self.ring.wait_until_empty(timeout)
        self.check_last_error()
        return drained

    def stage(
        self,
        *,
        store_key: str,
        tensors: Iterable[tuple[str, torch.Tensor]],
        run_id: str = "",
        attempt_id: str = "",
        data_id: str = "",
        request_id: str = "",
        source_engine: str = "",
    ) -> MooncakeP2PDescriptor:
        if self.role != "source" or self.ring is None or self.copy_stream is None:
            raise RuntimeError("Mooncake P2P source ring is not initialized")
        self.check_last_error()

        wait_start_ns = time.monotonic_ns()
        # A full source ring is normal consumer backpressure, not a failed
        # fetch. The controller's bounded optimizer/eval deadline fences stuck
        # read owners and releases their rights (or terminates this producer).
        # Reusing the read retry timeout here can kill a healthy shared Target
        # before the failed consumer's deadline, preventing that recovery.
        lease = self.ring.acquire(store_key)
        wait_end_ns = time.monotonic_ns()
        self._add_metric("ring_wait_seconds", (wait_end_ns - wait_start_ns) / 1e9)
        self._timeline_complete(
            "gpu_ring_wait",
            wait_start_ns,
            wait_end_ns,
            mooncake_key=store_key,
            slot_id=lease.slot_id,
            generation=lease.generation,
            ring_role="source",
            data_id=data_id,
            request_id=request_id,
            source_engine=source_engine,
        )

        buffer = self.buffers[lease.slot_id]
        descriptors: list[MooncakeP2PTensorDescriptor] = []
        offset = 0
        stage_start_ns = time.monotonic_ns()
        try:
            producer_stream = torch.cuda.current_stream(self.device)
            ready = torch.cuda.Event()
            ready.record(producer_stream)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(ready)
                for name, tensor in tensors:
                    if not tensor.is_cuda:
                        raise ValueError(f"P2P tensor {name} is not on CUDA")
                    tensor = tensor.contiguous()
                    offset = _align(offset)
                    length = tensor.numel() * tensor.element_size()
                    end = offset + length
                    if end > buffer.numel():
                        raise RuntimeError(
                            f"P2P payload exceeds slot capacity: need={end}, "
                            f"capacity={buffer.numel()}"
                        )
                    destination = buffer[offset:end].view(tensor.dtype).reshape(tensor.shape)
                    destination.copy_(tensor, non_blocking=True)
                    tensor.record_stream(self.copy_stream)
                    descriptors.append(
                        MooncakeP2PTensorDescriptor(
                            name=name,
                            shape=tuple(tensor.shape),
                            dtype=_dtype_name(tensor.dtype),
                            offset=offset,
                            length=length,
                        )
                    )
                    offset = end
                done = torch.cuda.Event()
                done.record(self.copy_stream)
            done.synchronize()
            descriptor = MooncakeP2PDescriptor(
                version=P2P_DESCRIPTOR_VERSION,
                run_id=run_id,
                attempt_id=attempt_id,
                store_key=store_key,
                data_id=data_id,
                request_id=request_id,
                source_engine=source_engine,
                source_endpoint=self.endpoint,
                source_release_endpoint=self.release_endpoint,
                slot_base_ptr=buffer.data_ptr(),
                slot_id=lease.slot_id,
                generation=lease.generation,
                payload_bytes=offset,
                tensors=tuple(descriptors),
            )
        except BaseException:
            # Copies queued before a later tensor/descriptor failure may still
            # write this slot. Retain its lease if draining itself fails.
            self.copy_stream.synchronize()
            self.ring.release(lease.slot_id, lease.generation, lease.owner)
            raise
        stage_end_ns = time.monotonic_ns()
        payload_bytes = offset
        self._timeline_complete(
            "gpu_stage",
            stage_start_ns,
            stage_end_ns,
            mooncake_key=store_key,
            payload_bytes=payload_bytes,
            slot_id=lease.slot_id,
            generation=lease.generation,
            stage_role="source_pack",
            data_id=data_id,
            request_id=request_id,
            source_engine=source_engine,
        )

        self._add_metric("source_descriptors")
        self._add_metric("source_payload_bytes", payload_bytes)
        return descriptor

    def reserve(
        self,
        *,
        store_key: str,
        tensors: Iterable[tuple[str, tuple[int, ...], torch.dtype]],
        run_id: str = "",
        attempt_id: str = "",
        data_id: str = "",
        request_id: str = "",
        source_engine: str = "",
        feature_layer_width: int | None = None,
    ) -> MooncakeP2PReservation:
        """Reserve registered source storage and return writable tensor views."""

        if self.role != "source" or self.ring is None:
            raise RuntimeError("Mooncake P2P source ring is not initialized")
        self.check_last_error()

        wait_start_ns = time.monotonic_ns()
        # Direct-prefill reservations have the same backpressure/owner-deadline
        # semantics as stage(); a fetch retry budget must not kill the source.
        lease = self.ring.acquire(store_key)
        wait_end_ns = time.monotonic_ns()
        self._add_metric("ring_wait_seconds", (wait_end_ns - wait_start_ns) / 1e9)
        self._timeline_complete(
            "gpu_ring_wait",
            wait_start_ns,
            wait_end_ns,
            mooncake_key=store_key,
            slot_id=lease.slot_id,
            generation=lease.generation,
            ring_role="source_direct",
            data_id=data_id,
            request_id=request_id,
            source_engine=source_engine,
        )

        buffer = self.buffers[lease.slot_id]
        try:
            layout, payload_bytes = _build_tensor_layout(tensors, capacity=buffer.numel())
            if feature_layer_width is not None:
                layout = tuple(replace(item, layer_width=feature_layer_width)
                               if item.name == "hidden_states" else item for item in layout)
            views = {
                item.name: buffer[item.offset : item.offset + item.length]
                .view(_torch_dtype(item.dtype))
                .reshape((item.shape[-1] // item.layer_width, item.shape[0], item.layer_width)
                         if item.layer_width is not None else item.shape)
                for item in layout
            }
            descriptor = MooncakeP2PDescriptor(
                version=P2P_DESCRIPTOR_VERSION,
                run_id=run_id,
                attempt_id=attempt_id,
                store_key=store_key,
                data_id=data_id,
                request_id=request_id,
                source_engine=source_engine,
                source_endpoint=self.endpoint,
                source_release_endpoint=self.release_endpoint,
                slot_base_ptr=buffer.data_ptr(),
                slot_id=lease.slot_id,
                generation=lease.generation,
                payload_bytes=payload_bytes,
                tensors=layout,
            )
        except BaseException:
            self.ring.release(lease.slot_id, lease.generation, lease.owner)
            raise
        if self.timeline is not None:
            self.timeline.event(
                "gpu_slot_reserve",
                mooncake_key=store_key,
                payload_bytes=payload_bytes,
                slot_id=lease.slot_id,
                generation=lease.generation,
                ring_role="source_direct",
                data_id=data_id,
                request_id=request_id,
                source_engine=source_engine,
            )
        return MooncakeP2PReservation(self, lease, descriptor, views)

    def _commit_reservation(self, reservation: MooncakeP2PReservation) -> None:
        if self.role != "source" or self.ring is None or self.device is None:
            raise RuntimeError("Mooncake P2P source ring is not initialized")
        self.check_last_error()
        start_ns = time.monotonic_ns()
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        done.synchronize()
        end_ns = time.monotonic_ns()
        descriptor = reservation.descriptor
        self._add_metric("source_descriptors")
        self._add_metric("source_payload_bytes", descriptor.payload_bytes)
        self._timeline_complete(
            "gpu_stage",
            start_ns,
            end_ns,
            mooncake_key=descriptor.store_key,
            payload_bytes=descriptor.payload_bytes,
            slot_id=descriptor.slot_id,
            generation=descriptor.generation,
            stage_role="source_direct_commit",
            data_id=descriptor.data_id,
            request_id=descriptor.request_id,
            source_engine=descriptor.source_engine,
        )

    def _discard_reservation(self, reservation: MooncakeP2PReservation) -> None:
        if self.role != "source" or self.ring is None:
            raise RuntimeError("Mooncake P2P source ring is not initialized")
        descriptor = reservation.descriptor
        # discard() holds the reservation lock. Uncommitted direct-output
        # writes may still run on a producer stream after a failed forward.
        # This is an abort path; committed output was already synchronized.
        if reservation._state == "reserved":
            torch.cuda.synchronize(self.device)
        self.ring.release(
            reservation._lease.slot_id,
            reservation._lease.generation,
            reservation._lease.owner,
        )
        self._add_metric("discarded_source_descriptors")
        self._add_metric(
            "discarded_source_payload_bytes", descriptor.payload_bytes
        )
        if self.timeline is not None:
            self.timeline.event(
                "gpu_slot_release",
                mooncake_key=descriptor.store_key,
                slot_id=descriptor.slot_id,
                generation=descriptor.generation,
                ring_role="source_direct_discard",
                data_id=descriptor.data_id,
                request_id=descriptor.request_id,
                source_engine=descriptor.source_engine,
            )

    def read(self, descriptor: MooncakeP2PDescriptor) -> MooncakeP2PReadResult:
        if self.role != "destination" or self.ring is None or self.copy_stream is None:
            raise RuntimeError("Mooncake P2P destination ring is not initialized")
        # CUDA's current device is host-thread local. PrefetchedDataFetcher calls
        # read() from a new worker thread, whose default is device 0 even on
        # nonzero DP ranks. Keep Mooncake's status polling and CUDA stream work
        # in the destination buffer's device context.
        torch.cuda.set_device(self.device)
        self.check_last_error()
        expected_run_id = os.getenv("ASPEC_RUN_ID", "")
        if expected_run_id and descriptor.run_id != expected_run_id:
            raise RuntimeError(
                "Mooncake P2P descriptor belongs to a stale or foreign run: "
                f"expected={expected_run_id!r}, observed={descriptor.run_id!r}"
            )
        expected_attempt_id = os.getenv("ASPEC_ATTEMPT_ID", "")
        if expected_attempt_id and descriptor.attempt_id != expected_attempt_id:
            raise RuntimeError(
                "Mooncake P2P descriptor belongs to a stale or foreign attempt: "
                f"expected={expected_attempt_id!r}, observed={descriptor.attempt_id!r}"
            )
        if descriptor.payload_bytes > int(self.config.gpu_buffer_size):
            raise RuntimeError(
                f"P2P payload {descriptor.payload_bytes} exceeds destination slot "
                f"capacity {self.config.gpu_buffer_size}"
            )

        descriptor_key = descriptor.to_key()
        wait_start_ns = time.monotonic_ns()
        timeout = self.config.get_retry_max_wait_seconds
        lease = self.ring.acquire(descriptor_key, timeout=timeout if timeout > 0 else None)
        wait_end_ns = time.monotonic_ns()
        self._add_metric("ring_wait_seconds", (wait_end_ns - wait_start_ns) / 1e9)
        self._timeline_complete(
            "gpu_ring_wait",
            wait_start_ns,
            wait_end_ns,
            mooncake_key=descriptor_key,
            slot_id=lease.slot_id,
            generation=lease.generation,
            ring_role="destination",
            data_id=descriptor.data_id,
            request_id=descriptor.request_id,
            source_engine=descriptor.source_engine,
        )

        buffer = self.buffers[lease.slot_id]
        notify_message = json.dumps(
            {
                "store_key": descriptor.store_key,
                "run_id": descriptor.run_id,
                "attempt_id": descriptor.attempt_id,
                "slot_id": descriptor.slot_id,
                "generation": descriptor.generation,
                "data_id": descriptor.data_id,
                "request_id": descriptor.request_id,
                "source_engine": descriptor.source_engine,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            from mooncake.engine import TransferNotify, TransferOpcode

            transfer_start_ns = time.monotonic_ns()
            notify = (
                None
                if self.config.protocol in {"tcp", "rdma"} or descriptor.reader_id
                else TransferNotify(P2P_RELEASE_NOTIFY, notify_message)
            )
            try:
                rc, transfer_chunks = self._read_payload(
                    descriptor,
                    destination_ptr=buffer.data_ptr(),
                    opcode=TransferOpcode.Read,
                    notify=notify,
                )
            except BaseException:
                self._add_metric("transfer_failures")
                raise
            transfer_end_ns = time.monotonic_ns()
            transfer_seconds = (transfer_end_ns - transfer_start_ns) / 1e9
            self._timeline_complete(
                "mooncake_p2p_read",
                transfer_start_ns,
                transfer_end_ns,
                mooncake_key=descriptor_key,
                payload_bytes=descriptor.payload_bytes,
                source_endpoint=descriptor.source_endpoint,
                source_slot_id=descriptor.slot_id,
                source_generation=descriptor.generation,
                destination_slot_id=lease.slot_id,
                transfer_rc=rc,
                transfer_chunks=transfer_chunks,
                data_id=descriptor.data_id,
                request_id=descriptor.request_id,
                source_engine=descriptor.source_engine,
            )
            if rc != 0:
                self._add_metric("transfer_failures")
                raise RuntimeError(f"Mooncake P2P READ failed (rc={rc})")
            if self.config.protocol in {"tcp", "rdma"} or descriptor.reader_id:
                # The acknowledged release proves that the process owning the
                # registered source generation is alive and has accepted the
                # ownership transition.  Mooncake's optional READ notify does
                # not expose delivery failure through the Python return code.
                self._acknowledge_source_release(descriptor)
            self._add_metric("destination_reads")
            self._add_metric("destination_transfer_chunks", transfer_chunks)
            self._add_metric("destination_payload_bytes", descriptor.payload_bytes)
            self._add_metric("destination_transfer_seconds", transfer_seconds)
            with self._metrics_lock:
                self._metrics["destination_max_transfer_seconds"] = max(
                    float(self._metrics["destination_max_transfer_seconds"]),
                    transfer_seconds,
                )

            result: dict[str, torch.Tensor] = {}
            for tensor in descriptor.tensors:
                dtype = _torch_dtype(tensor.dtype)
                numel = tensor.length // torch.empty((), dtype=dtype).element_size()
                value = (
                    buffer[tensor.offset : tensor.offset + tensor.length]
                    .view(dtype)[:numel]
                    .reshape(tensor.shape)
                )
                if tensor.layer_width is not None:
                    value = value.reshape(tensor.shape[-1] // tensor.layer_width,
                                          tensor.shape[0], tensor.layer_width)
                    value = value.permute(1, 0, 2).reshape(tensor.shape)
                result[tensor.name] = value
            return MooncakeP2PReadResult(
                tensors=result,
                lease=MooncakeP2PReceiveLease(self, lease, descriptor),
            )
        except BaseException:
            self.ring.release(lease.slot_id, lease.generation, lease.owner)
            raise

    def discard_staged(self, descriptor: MooncakeP2PDescriptor) -> None:
        """Release an unpublished source descriptor after a multi-shard failure."""

        if self.role != "source" or self.ring is None:
            raise RuntimeError("Mooncake P2P source ring is not initialized")
        self.ring.release(
            descriptor.slot_id,
            descriptor.generation,
            descriptor.store_key,
        )
        self._add_metric("discarded_source_descriptors")
        self._add_metric(
            "discarded_source_payload_bytes", descriptor.payload_bytes
        )
        if self.timeline is not None:
            self.timeline.event(
                "gpu_slot_release",
                mooncake_key=descriptor.store_key,
                slot_id=descriptor.slot_id,
                generation=descriptor.generation,
                ring_role="source_discard",
                data_id=descriptor.data_id,
                request_id=descriptor.request_id,
                source_engine=descriptor.source_engine,
            )

    def _release_destination_lease(
        self, lease: RingLease, descriptor: MooncakeP2PDescriptor
    ) -> None:
        if self.ring is None:
            raise RuntimeError("Mooncake P2P destination ring is closed")
        self.ring.release(lease.slot_id, lease.generation, lease.owner)
        self._add_metric("destination_releases")
        if self.timeline is not None:
            self.timeline.event(
                "gpu_slot_release",
                mooncake_key=descriptor.to_key(),
                slot_id=lease.slot_id,
                generation=lease.generation,
                ring_role="destination",
                data_id=descriptor.data_id,
                request_id=descriptor.request_id,
                source_engine=descriptor.source_engine,
            )

    def close(self) -> None:
        self._stop.set()
        if self._release_server is not None:
            try:
                self._release_server.close()
            finally:
                self._release_server = None
        if self._release_thread is not None:
            release_thread = self._release_thread
            release_thread.join(timeout=2.0)
            if release_thread.is_alive():
                error = RuntimeError(
                    "Mooncake P2P release server did not stop within 2 seconds"
                )
                with self._error_lock:
                    self._last_error = self._last_error or error
                raise error
            self._release_thread = None
        if self._notify_thread is not None:
            notify_thread = self._notify_thread
            notify_thread.join(timeout=2.0)
            if notify_thread.is_alive():
                error = RuntimeError(
                    "Mooncake P2P notification worker did not stop within 2 seconds"
                )
                with self._error_lock:
                    self._last_error = self._last_error or error
                raise error
            self._notify_thread = None
        if self.engine is not None:
            unregister_errors = []
            for buffer in ([] if self._ring_buffer is None else [self._ring_buffer]):
                try:
                    rc = self.engine.unregister_memory(buffer.data_ptr())
                    if rc not in (None, 0):
                        unregister_errors.append(
                            f"ptr={buffer.data_ptr():#x}, rc={rc}"
                        )
                except Exception as exc:
                    unregister_errors.append(
                        f"ptr={buffer.data_ptr():#x}, error={type(exc).__name__}: {exc}"
                    )
            if unregister_errors:
                raise RuntimeError(
                    "Mooncake P2P GPU memory unregister failed: "
                    + "; ".join(unregister_errors)
                )
        self.buffers.clear()
        self._ring_buffer = None
        self.copy_stream = None
        self.engine = None

    def _poll_notifies(self) -> None:
        try:
            while not self._stop.is_set():
                notifications = self.engine.get_notifies()
                if not notifications:
                    self._stop.wait(0.01)
                    continue
                for notification in notifications:
                    if notification.name != P2P_RELEASE_NOTIFY:
                        continue
                    value = json.loads(notification.msg)
                    expected_run_id = os.getenv("ASPEC_RUN_ID", "")
                    expected_attempt_id = os.getenv("ASPEC_ATTEMPT_ID", "")
                    if expected_run_id and value.get("run_id") != expected_run_id:
                        raise RuntimeError(
                            "Mooncake P2P release belongs to a stale or foreign run"
                        )
                    if (
                        expected_attempt_id
                        and value.get("attempt_id") != expected_attempt_id
                    ):
                        raise RuntimeError(
                            "Mooncake P2P release belongs to a stale or foreign attempt"
                        )
                    slot_id = int(value["slot_id"])
                    generation = int(value["generation"])
                    store_key = str(value["store_key"])
                    self.ring.release(slot_id, generation, store_key)
                    self._add_metric("source_releases")
                    if self.timeline is not None:
                        self.timeline.event(
                            "gpu_slot_release",
                            mooncake_key=store_key,
                            slot_id=slot_id,
                            generation=generation,
                            ring_role="source",
                            data_id=value.get("data_id"),
                            request_id=value.get("request_id"),
                            source_engine=value.get("source_engine"),
                        )
        except BaseException as exc:
            with self._error_lock:
                self._last_error = exc
            logger.exception("Mooncake P2P notification worker failed")

    def _timeline_complete(self, name: str, start_ns: int, end_ns: int, **fields: Any) -> None:
        if self.timeline is not None:
            self.timeline.complete(name, start_ns=start_ns, end_ns=end_ns, **fields)
