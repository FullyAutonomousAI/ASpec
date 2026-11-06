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

"""Bounded, checksum-verified checkpoint exchange for replicated local filesystems.

Reimplemented for ASpec's explicit node-local storage contract: the retained
checkpoint implementation assumes shared filesystem visibility. Ray carries file
chunks; DCP still owns tensor/shard layout and checkpoint publication is unchanged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

CHUNK_BYTES = 8 * 1024 * 1024
_STORES = {}


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint exchange rejects symlinks: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": file_digest(path),
            }
    return result


class LocalFiles:
    """One node's task-owned output tree. Never reads/writes arbitrary worker paths."""

    def __init__(self, output, run_id):
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.root / ".node-local-owner"
        if owner.exists() and owner.read_text() != run_id:
            raise ValueError(f"Output belongs to another run: {self.root}")
        # Driver may have already created plan/evidence. Existing remote training
        # artifacts must not be silently reused by another attempt.
        if not owner.exists() and any(
            (self.root / name).exists() for name in ("checkpoints", "resume-input")
        ):
            raise ValueError(f"Use a fresh node-local output: {self.root}")
        owner.write_text(run_id)

    def path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Exchange path must stay inside the run directory")
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Exchange path escapes the run directory")
        return path

    def inventory(self, relative):
        return manifest(self.path(relative))

    def read(self, relative, offset):
        if offset < 0:
            raise ValueError("Negative offset")
        with self.path(relative).open("rb") as stream:
            stream.seek(offset)
            return stream.read(CHUNK_BYTES)

    def append(self, relative, offset, data):
        if len(data) > CHUNK_BYTES or offset < 0:
            raise ValueError("Invalid chunk")
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".exchange-partial")
        if offset == 0:
            with temporary.open("xb") as stream:
                stream.write(data)
        else:
            if temporary.stat().st_size != offset:
                raise ValueError("Out-of-order checkpoint chunk")
            with temporary.open("ab") as stream:
                stream.write(data)

    def publish(self, relative, expected):
        target = self.path(relative)
        temporary = target.with_name(target.name + ".exchange-partial")
        if (
            temporary.stat().st_size != expected["size"]
            or file_digest(temporary) != expected["sha256"]
        ):
            raise ValueError("Checkpoint transfer checksum mismatch")
        if target.exists():
            if file_digest(target) != expected["sha256"]:
                raise ValueError("Refusing conflicting checkpoint bytes")
            temporary.unlink()
        else:
            os.replace(temporary, target)

    def remove_staging(self, relative):
        path = self.path(relative)
        if (
            path.parent.name != "checkpoints"
            or not path.name.startswith(".iter_")
            or not path.name.endswith(".incomplete")
        ):
            raise ValueError("Only unpublished checkpoint staging may be removed")
        shutil.rmtree(path, ignore_errors=False)


def enabled(args):
    return (
        getattr(args, "cluster_mode", "single_node") == "static_multi_node"
        and getattr(args, "cluster_storage_mode", "shared") == "node_local"
    )


def stores(args):
    key = (str(args.output_dir), str(args.aspec_run_id))
    if key not in _STORES:
        ips = set(args.training_node_ips) | set(args.inference_node_ips)
        nodes = {n["NodeManagerAddress"]: n for n in ray.nodes() if n.get("Alive")}
        if not ips <= set(nodes):
            raise ValueError("A planned node is unavailable for checkpoint exchange")
        actor = ray.remote(num_cpus=0)(LocalFiles)
        _STORES[key] = {
            ip: actor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip]["NodeID"], soft=False)
            ).remote(args.output_dir, args.aspec_run_id)
            for ip in sorted(ips)
        }
    return _STORES[key]


def wait(args, ref):
    return ray.get(ref, timeout=float(getattr(args, "cluster_file_transfer_timeout_seconds", 600)))


def send_tree(args, source, relative):
    files = manifest(source)
    for actor in stores(args).values():
        for name, info in files.items():
            with (Path(source) / name).open("rb") as stream:
                offset = 0
                while True:
                    chunk = stream.read(CHUNK_BYTES)
                    if not chunk and offset:
                        break
                    wait(args, actor.append.remote(str(Path(relative) / name), offset, chunk))
                    offset += len(chunk)
                    if not chunk:
                        break
            wait(args, actor.publish.remote(str(Path(relative) / name), info))


def prepare_resume(args):
    if not enabled(args):
        return
    for actor in stores(args).values():
        wait(args, actor.inventory.remote("."))
    if getattr(args, "load_path", None):
        from aspec.training.checkpoint import resolve_resume_model_dir

        source = resolve_resume_model_dir(args)
        if source is None:
            raise ValueError("Coordinator cannot resolve the requested checkpoint")
        source = source.parent
        args.resume_source_run_dir = str(source.parent.parent)
        relative = Path("resume-input") / source.name
        shutil.copytree(source, Path(args.output_dir) / relative)
        send_tree(args, source, relative)
        args.load_path = str(Path(args.output_dir) / relative)


def collect_checkpoint(args, completed_step):
    if not enabled(args):
        return
    relative = Path("checkpoints") / f".iter_{completed_step + 1:07d}.incomplete"
    destination = Path(args.output_dir) / relative
    destination.mkdir(parents=True, exist_ok=True)
    for ip, actor in stores(args).items():
        if ip not in args.training_node_ips:
            continue
        files = wait(args, actor.inventory.remote(str(relative)))
        for name, info in files.items():
            target = destination / name
            if target.exists():
                if target.stat().st_size != info["size"] or file_digest(target) != info["sha256"]:
                    raise ValueError(f"Conflicting checkpoint shard: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".collect-partial")
            with temporary.open("xb") as stream:
                for offset in range(0, info["size"], CHUNK_BYTES):
                    chunk = wait(args, actor.read.remote(str(relative / name), offset))
                    if len(chunk) != min(CHUNK_BYTES, info["size"] - offset):
                        raise ValueError("Truncated checkpoint exchange")
                    stream.write(chunk)
            if file_digest(temporary) != info["sha256"]:
                raise ValueError("Collected checkpoint checksum mismatch")
            os.replace(temporary, target)


def clear_staging(args, completed_step):
    if not enabled(args):
        return
    relative = Path("checkpoints") / f".iter_{completed_step + 1:07d}.incomplete"
    # The coordinator's tree was atomically renamed during publication. Missing
    # staging on its node is expected; remote unpublished staging is disposable.
    for ip, actor in stores(args).items():
        if ip in args.training_node_ips:
            try:
                wait(args, actor.remove_staging.remote(str(relative)))
            except ray.exceptions.RayTaskError as error:
                if not isinstance(error.as_instanceof_cause(), FileNotFoundError):
                    raise


def close_stores(args):
    key = (str(args.output_dir), str(getattr(args, "aspec_run_id", "")))
    for actor in _STORES.pop(key, {}).values():
        if ray.is_initialized():
            ray.kill(actor, no_restart=True)
