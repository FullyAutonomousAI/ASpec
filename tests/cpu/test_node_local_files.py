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

from types import SimpleNamespace

import pytest

from aspec.ray import node_local_files as files


class RemoteMethod:
    def __init__(self, method):
        self.remote = method


class FakeActor:
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.store, name))


def test_chunk_exchange_checks_order_hash_and_conflicts(tmp_path):
    node = files.LocalFiles(tmp_path / "worker", "run")
    data = b"checkpoint"
    import hashlib

    info = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    node.append("resume/model.bin", 0, data[:3])
    with pytest.raises(ValueError, match="Out-of-order"):
        node.append("resume/model.bin", 2, data[3:])
    node.append("resume/model.bin", 3, data[3:])
    with pytest.raises(ValueError, match="checksum"):
        node.publish("resume/model.bin", {"size": 1, "sha256": "bad"})
    node.publish("resume/model.bin", info)
    assert node.read("resume/model.bin", 0) == data
    with pytest.raises(ValueError):
        node.path("../outside")
    with pytest.raises(ValueError):
        node.path("/outside")
    with pytest.raises(ValueError, match="another run"):
        files.LocalFiles(tmp_path / "worker", "other")


def test_collects_disjoint_shards_without_shared_directory(tmp_path, monkeypatch):
    output = tmp_path / "coordinator"
    output.mkdir()
    args = SimpleNamespace(
        cluster_mode="static_multi_node",
        cluster_storage_mode="node_local",
        output_dir=str(output),
        training_node_ips=["a", "b"],
    )
    actors = {}
    for ip in ("a", "b"):
        node = files.LocalFiles(tmp_path / ip, "run")
        shard = node.path(f"checkpoints/.iter_0000005.incomplete/model/{ip}.distcp")
        shard.parent.mkdir(parents=True)
        shard.write_bytes(ip.encode() * 10)
        actors[ip] = FakeActor(node)
    monkeypatch.setattr(files, "stores", lambda _: actors)
    monkeypatch.setattr(files, "wait", lambda args, ref: ref)
    files.collect_checkpoint(args, 4)
    target = output / "checkpoints/.iter_0000005.incomplete/model"
    assert (target / "a.distcp").read_bytes() == b"a" * 10
    assert (target / "b.distcp").read_bytes() == b"b" * 10
    assert not (tmp_path / "a/checkpoints/.iter_0000005.incomplete/model/b.distcp").exists()
    # A conflicting same-name shard cannot silently overwrite a previously collected file.
    actors["b"].store.path("checkpoints/.iter_0000005.incomplete/model/a.distcp").write_bytes(
        b"wrong"
    )
    with pytest.raises(ValueError, match="Conflicting"):
        files.collect_checkpoint(args, 4)


def test_resume_replication_rejects_partial_and_preserves_bytes(tmp_path, monkeypatch):
    source = tmp_path / "original/iter_0000005"
    source.mkdir(parents=True)
    (source / "meta.json").write_text("{}")
    (source / "empty").write_bytes(b"")
    actors = {ip: FakeActor(files.LocalFiles(tmp_path / ip, "run")) for ip in ("a", "b")}
    args = SimpleNamespace(output_dir=str(tmp_path / "coordinator"))
    monkeypatch.setattr(files, "stores", lambda _: actors)
    monkeypatch.setattr(files, "wait", lambda args, ref: ref)
    files.send_tree(args, source, "resume-input/iter_0000005")
    for actor in actors.values():
        assert files.manifest(actor.store.path("resume-input/iter_0000005")) == files.manifest(
            source
        )
        assert not list(actor.store.root.rglob("*.exchange-partial"))


def test_manifest_rejects_symlink_and_remove_is_staging_only(tmp_path):
    node = files.LocalFiles(tmp_path / "worker", "run")
    node.path("link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        node.inventory(".")
    with pytest.raises(ValueError, match="Only unpublished"):
        node.remove_staging("checkpoints/iter_0000005")
