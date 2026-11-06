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

"""Low-overhead, process-local timeline events for end-to-end experiments.

All ASpec actors in a single-node run share Linux CLOCK_MONOTONIC, so
``time.monotonic_ns()`` timestamps can be correlated without wall-clock skew.
Each process writes a separate JSONL file to avoid inter-process locking.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_RECORDERS: dict[tuple[str, str, str, str, int], "TimelineRecorder"] = {}
_RECORDERS_LOCK = threading.Lock()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _safe_name(value: Any) -> str:
    return _SAFE_NAME.sub("_", str(value)).strip("_") or "unknown"


def tensor_sha256(value: Any) -> str:
    """Return a byte-exact digest for a tensor-like payload.

    This helper is only called by instrumented correctness runs.  Keeping it
    out of the normal data path makes the disabled timeline a true no-op.
    """
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


class TimelineRecorder:
    """Buffered JSONL writer. Disabled recorders are true no-ops."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_dir: str = "",
        run_id: str = "",
        role: str = "process",
        rank: int | str = 0,
        flush_every: int = 128,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = str(output_dir or "")
        self.run_id = str(run_id or "")
        self.role = str(role)
        self.rank = rank
        self.flush_every = max(1, int(flush_every))
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self._lock = threading.Lock()
        self._event_count = 0
        self._stream = None
        self.path: Path | None = None

        if not self.enabled:
            return
        if not self.output_dir:
            raise ValueError("timeline_output_dir is required when timeline_enabled=true")
        if not self.run_id:
            raise ValueError("timeline_run_id is required when timeline_enabled=true")

        run_dir = Path(self.output_dir).expanduser() / self.run_id / "events"
        run_dir.mkdir(parents=True, exist_ok=True)
        filename = ".".join(
            _safe_name(part)
            for part in (self.hostname, self.role, self.rank, self.pid)
        )
        self.path = run_dir / f"{filename}.jsonl"
        self._stream = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)
        atexit.register(self.close)
        self.event("process_start")

    def event(self, name: str, **fields: Any) -> None:
        """Record an instant event at the current monotonic timestamp."""
        if not self.enabled:
            return
        self._write(name=name, phase="i", ts_ns=time.monotonic_ns(), fields=fields)

    def complete(
        self,
        name: str,
        *,
        start_ns: int,
        end_ns: int | None = None,
        **fields: Any,
    ) -> None:
        """Record a complete (Chrome ``X`` phase) interval."""
        if not self.enabled:
            return
        end_ns = time.monotonic_ns() if end_ns is None else int(end_ns)
        start_ns = int(start_ns)
        self._write(
            name=name,
            phase="X",
            ts_ns=start_ns,
            fields={"end_ns": end_ns, "duration_ns": end_ns - start_ns, **fields},
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        nvtx: bool = False,
        nvtx_name: str | None = None,
        **fields: Any,
    ) -> Iterator[None]:
        """Record a complete interval and optionally mirror it to an NVTX range."""
        # A disabled timeline is a strict no-op. Plain throughput runs therefore
        # do not pay for NVTX imports/ranges; the dedicated Timeline/Nsight runs
        # enable the recorder and receive both JSONL and NVTX ranges.
        if not self.enabled:
            yield
            return

        pushed = False
        if nvtx:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(nvtx_name or name)
                    pushed = True
            except Exception:
                pushed = False

        start_ns = time.monotonic_ns()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end_ns = time.monotonic_ns()
            if pushed:
                try:
                    import torch

                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
            if self.enabled:
                if error is not None:
                    fields = {**fields, "error": error}
                self.complete(name, start_ns=start_ns, end_ns=end_ns, **fields)

    def flush(self) -> None:
        if self._stream is None:
            return
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            self.event("process_stop")
            self.flush()
        finally:
            with self._lock:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None

    def _write(self, *, name: str, phase: str, ts_ns: int, fields: dict[str, Any]) -> None:
        stream = self._stream
        if stream is None:
            return
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "event": str(name),
            "phase": phase,
            "ts_ns": int(ts_ns),
            "role": self.role,
            "rank": self.rank,
            "pid": self.pid,
            "thread_id": threading.get_ident(),
            "hostname": self.hostname,
            **{key: value for key, value in fields.items() if value is not None},
        }
        payload = json.dumps(event, default=_json_default, separators=(",", ":"))
        with self._lock:
            if self._stream is None:
                return
            self._stream.write(payload + "\n")
            self._event_count += 1
            if self._event_count % self.flush_every == 0:
                self._stream.flush()


def get_timeline(args: Any, role: str, rank: int | str = 0) -> TimelineRecorder:
    """Return one recorder per process/output/run/role/rank tuple."""
    enabled = bool(getattr(args, "timeline_enabled", False))
    output_dir = str(getattr(args, "timeline_output_dir", "") or "")
    run_id = str(getattr(args, "timeline_run_id", "") or "")
    flush_every = int(getattr(args, "timeline_flush_every", 128) or 128)
    key = (output_dir, run_id, str(role), str(rank), os.getpid())
    with _RECORDERS_LOCK:
        recorder = _RECORDERS.get(key)
        if recorder is None:
            recorder = TimelineRecorder(
                enabled=enabled,
                output_dir=output_dir,
                run_id=run_id,
                role=role,
                rank=rank,
                flush_every=flush_every,
            )
            _RECORDERS[key] = recorder
        return recorder


def export_timeline_env(args: Any) -> None:
    """Export trace settings for SGLang scheduler subprocesses."""
    os.environ["ASPEC_TIMELINE_ENABLED"] = (
        "1" if bool(getattr(args, "timeline_enabled", False)) else "0"
    )
    os.environ["ASPEC_TIMELINE_OUTPUT_DIR"] = str(
        getattr(args, "timeline_output_dir", "") or ""
    )
    os.environ["ASPEC_TIMELINE_RUN_ID"] = str(getattr(args, "timeline_run_id", "") or "")
    os.environ["ASPEC_TIMELINE_FLUSH_EVERY"] = str(
        int(getattr(args, "timeline_flush_every", 128) or 128)
    )


def get_timeline_from_env(role: str, rank: int | str = 0) -> TimelineRecorder:
    """Build a recorder inside a subprocess that only inherited environment."""

    class _Args:
        timeline_enabled = os.getenv("ASPEC_TIMELINE_ENABLED", "0") == "1"
        timeline_output_dir = os.getenv("ASPEC_TIMELINE_OUTPUT_DIR", "")
        timeline_run_id = os.getenv("ASPEC_TIMELINE_RUN_ID", "")
        timeline_flush_every = int(os.getenv("ASPEC_TIMELINE_FLUSH_EVERY", "128"))

    return get_timeline(_Args(), role=role, rank=rank)
