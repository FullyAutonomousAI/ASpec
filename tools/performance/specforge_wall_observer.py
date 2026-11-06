# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Observation-only hooks for pinned SpecForge 03f3682's native training loop.

Loaded from an isolated bootstrap directory, never a replacement framework checkout.
Hooks delegate to the original loader/controller and do not touch tensor values,
CUDA streams, RNG, optimization, leases or collectives. Ref metadata supplies exact
input lengths without a GPU reduction or device-to-host synchronization.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import socket
import sys
import time
from pathlib import Path

_TOKEN_KEY = "_aspec_benchmark_input_tokens"


def patch_loader(cls):
    original = cls._make_batch

    @functools.wraps(original)
    def make_batch(self, refs):
        batch = original(self, refs)
        lengths = [ref.num_tokens for ref in refs]
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths):
            raise ValueError("Benchmark requires positive resident SampleRef.num_tokens")
        batch.metadata[_TOKEN_KEY] = sum(lengths)
        return batch

    cls._make_batch = make_batch


class _ObservedData:
    def __init__(self, data, state):
        self.data, self.state = data, state

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __iter__(self):
        iterator = iter(self.data)
        try:
            for batch in iterator:
                self.state["input_tokens"] += batch.metadata[_TOKEN_KEY]
                self.state["sample_ids"].extend(batch.sample_ids)
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


def patch_controller(cls):
    original = cls._fit

    @functools.wraps(original)
    def fit(self, data, progress):
        rank = int(os.environ["RANK"])
        path = Path(os.environ["ASPEC_SPECFORGE_OBSERVATION_DIR"]) / f"rank-{rank}.jsonl"
        state = {"input_tokens": 0, "sample_ids": []}
        logger = self.logger
        if logger is None or self.log_interval != 1:
            raise ValueError("Benchmark requires the native per-optimizer-step logger")
        with path.open("x", encoding="utf-8") as stream:
            def emit(row):
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()

            def observe(metrics, step):
                if "perf/optimizer_step_time_s" in metrics:
                    # Native metrics are already materialized here. Add no CUDA fence.
                    stamp = time.perf_counter_ns()
                    emit({"event": "optimizer_complete", "rank": rank, "step": step,
                          "clock_ns": stamp, "input_tokens": state["input_tokens"],
                          "sample_ids": state["sample_ids"]})
                    state["input_tokens"] = 0
                    state["sample_ids"] = []
                return logger(metrics, step)

            self.logger = observe
            try:
                emit({"event": "loop_start", "rank": rank, "hostname": socket.gethostname(),
                      "clock": "perf_counter_ns", "clock_ns": time.perf_counter_ns(),
                      "python": sys.executable, "controller_source": original.__code__.co_filename})
                return original(self, _ObservedData(data, state), progress)
            finally:
                self.logger = logger

    cls._fit = fit


class _Loader:
    def __init__(self, loader, patch, class_name):
        self.loader, self.patch, self.class_name = loader, patch, class_name

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.patch(getattr(module, self.class_name))

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _Finder(importlib.abc.MetaPathFinder):
    targets = {
        "specforge.training.controller": (patch_controller, "TrainerController"),
        "specforge.runtime.data_plane.feature_dataloader": (patch_loader, "FeatureDataLoader"),
    }

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, *self.targets[fullname])
        return spec


def install():
    if not os.environ.get("ASPEC_SPECFORGE_OBSERVATION_DIR"):
        return
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
