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
from unittest.mock import MagicMock

import pytest


def test_multi_draft_probe_is_closed_if_trainer_kill_fails(monkeypatch):
    from aspec import multi_draft as module

    actor, probe = object(), object()
    creator = MagicMock()
    creator.options.return_value.remote.return_value = probe
    monkeypatch.setattr(module, "_OwnerFenceProbe", creator)
    monkeypatch.setattr(
        module.ray,
        "nodes",
        lambda: [{"Alive": True, "NodeID": "1" * 56, "NodeManagerAddress": "ip"}],
    )
    killed = []

    def kill(target, **kw):
        killed.append(target)
        if target is actor:
            raise RuntimeError("kill dispatch failed")

    monkeypatch.setattr(module.ray, "kill", kill)
    with pytest.raises(RuntimeError, match="kill dispatch failed"):
        module._fence_group(SimpleNamespace(_actor_handlers=[actor]), [{"node_ip": "ip", "pid": 1}])
    assert probe in killed


def test_materializer_startup_failure_enters_cleanup(monkeypatch, tmp_path):
    from aspec.offline import generate as module

    args = SimpleNamespace(
        inference_engine_type="sgl",
        mooncake_data_plane="store",
        mooncake_protocol="tcp",
        draft_context_parallel_size=1,
        dflash_packing=False,
        last_hidden_states_prenorm=True,
        aspec_feature_contract={},
        target_model_path="target",
    )
    controller = MagicMock()
    factory = MagicMock()
    factory.options.return_value.remote.return_value = controller
    monkeypatch.setattr(module, "_get_draft_model_config", lambda a: SimpleNamespace(vocab_size=32))
    monkeypatch.setattr(module, "_validate_and_configure_dflash", lambda *a: None)
    monkeypatch.setattr(module, "OfflineDataset", MagicMock())
    monkeypatch.setattr(module, "target_identity", lambda p: {})
    monkeypatch.setattr(module, "_ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(module, "get_aspec_env_vars", lambda: {})
    monkeypatch.setattr(module, "AsyncTrainingController", factory)
    monkeypatch.setattr(
        module.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "1" * 56)
    )

    def fail(*a, **kw):
        raise RuntimeError("dataset load failed")

    monkeypatch.setattr(module.ray, "get", fail)
    monkeypatch.setattr(module.ray, "kill", MagicMock())
    cleanup = MagicMock(return_value={"state": "completed"})
    monkeypatch.setattr(module, "_safe_training_cleanup", cleanup)
    with pytest.raises(RuntimeError, match="dataset load failed"):
        module.materialize(args, str(tmp_path))
    cleanup.assert_called_once()
    assert cleanup.call_args.kwargs["controller"] is controller


@pytest.mark.parametrize("fence_ok", [False, True])
def test_saver_failure_fences_before_source_cleanup(monkeypatch, fence_ok):
    from aspec.offline import generate as module

    saver, manager, controller, engine, pg = [MagicMock() for _ in range(5)]
    events = []
    owner = {"node_id": "1" * 56, "node_ip": "local", "pid": 123}
    monkeypatch.setattr(module.ray, "get", MagicMock(side_effect=TimeoutError("reader blocked")))
    monkeypatch.setattr(module.ray, "kill", lambda actor, **kw: events.append(("kill", actor)))

    def fence(statuses, **kwargs):
        assert statuses == [owner]
        events.append(("fence", saver))
        if not fence_ok:
            raise TimeoutError("owner still alive")
        return {"state": "verified"}

    monkeypatch.setattr(module, "wait_for_owner_exit", fence)
    cleanup = MagicMock(
        side_effect=lambda *a, **k: events.append(("sources", engine)) or {"state": "completed"}
    )
    monkeypatch.setattr(module, "_safe_training_cleanup", cleanup)
    monkeypatch.setattr(
        module.ray.util, "remove_placement_group", lambda group: events.append(("pg", group))
    )
    result = module._cleanup_materialization(
        SimpleNamespace(),
        controller,
        manager,
        None,
        [engine],
        saver,
        owner,
        (pg, [0], [0]),
    )
    assert result["state"] == "completed_with_errors"
    if fence_ok:
        assert (
            events.index(("kill", saver))
            < events.index(("fence", saver))
            < events.index(("sources", engine))
            < events.index(("pg", pg))
        )
        assert cleanup.call_args.kwargs["closed_reader_fence"]["state"] == "verified"
    else:
        cleanup.assert_not_called()
        assert ("pg", pg) not in events
        assert ("kill", engine) not in events
        assert result["source_teardown"] == "blocked"


@pytest.mark.parametrize("source_state", ["completed", "completed_with_errors"])
def test_materialization_releases_only_the_placement_handle_after_source_cleanup(
    monkeypatch, source_state
):
    from aspec.offline import generate as module

    pg = object()
    events = []
    monkeypatch.setattr(
        module, "_safe_training_cleanup",
        lambda *a, **k: events.append("sources") or {"state": source_state},
    )
    remove = MagicMock(side_effect=lambda group: events.append("placement"))
    monkeypatch.setattr(module.ray.util, "remove_placement_group", remove)
    # create_placement_groups returns (handle, bundle indices, device ids).
    result = module._cleanup_materialization(
        SimpleNamespace(), None, None, None, [], None, None, (pg, [0], [2])
    )
    assert result["state"] == source_state
    if source_state == "completed":
        remove.assert_called_once_with(pg)
        assert events == ["sources", "placement"]
    else:
        remove.assert_not_called()


def test_main_disconnects_job_actors_after_initialization_error(monkeypatch, tmp_path):
    from aspec import train_entry as module
    from aspec.ray import node_local_files

    args = SimpleNamespace(output_dir=str(tmp_path), preflight_only=False)
    recorder = MagicMock()
    monkeypatch.setattr(module, "_prepare_config", lambda: (args, MagicMock()))
    monkeypatch.setattr(module, "_prepare_run_directory", lambda *a: None)
    monkeypatch.setattr(module, "RunEvidenceRecorder", lambda *a, **k: recorder)
    monkeypatch.setattr(module, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(module, "run_cluster_preflight", lambda a: None)
    monkeypatch.setattr(
        module, "train_async_no_generation", MagicMock(side_effect=ValueError("actor init failed"))
    )
    monkeypatch.setattr(module.ray, "is_initialized", lambda: True)
    shutdown = MagicMock()
    monkeypatch.setattr(module.ray, "shutdown", shutdown)
    monkeypatch.setattr(node_local_files, "close_stores", MagicMock())
    with pytest.raises(ValueError, match="actor init failed"):
        module.main()
    shutdown.assert_called_once()
    recorder.fail.assert_called_once()
    recorder.complete.assert_not_called()


@pytest.mark.parametrize(
    "fence_fails,observer_fails",
    [(False, False), (True, False), (True, True), (True, "cancel")],
)
def test_main_respects_training_reader_fence(monkeypatch, tmp_path, fence_fails, observer_fails):
    from aspec import train_entry
    from aspec.controller import loop
    from aspec.ray import node_local_files, owner_fence

    args = SimpleNamespace(output_dir=str(tmp_path), preflight_only=False)
    recorder = MagicMock()
    if observer_fails:
        recorder.record_cleanup.side_effect = OSError("evidence disk full")
        recorder.fail.side_effect = (
            KeyboardInterrupt("report interrupted") if observer_fails == "cancel"
            else OSError("evidence disk full")
        )
    trainer, source, manager, controller = [MagicMock() for _ in range(4)]
    owners = [{"node_id": "node", "node_ip": "ip", "pid": 123}]
    group = SimpleNamespace(_actor_handlers=[trainer], _owner_statuses=owners)
    controller.finalize_uncommitted_work.remote.return_value = {"discarded": 0}
    controller.get_work_lifecycle_status.remote.return_value = {
        "state": "incomplete", "resident_outputs": 1, "resident_bytes": 128,
    }
    controller.finalize_fenced_work.remote.return_value = {
        "discarded": 1,
        "lifecycle": {"state": "validated", "resident_outputs": 0, "resident_bytes": 0},
    }
    failure = ValueError("trainer failed")
    monkeypatch.setattr(train_entry, "_prepare_config", lambda: (args, object()))
    monkeypatch.setattr(train_entry, "RunEvidenceRecorder", lambda *a, **kw: recorder)
    monkeypatch.setattr(train_entry, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(train_entry, "run_cluster_preflight", lambda a: None)
    monkeypatch.setattr(train_entry, "install_termination_signal_handler", lambda: None)
    close_stores = MagicMock()
    monkeypatch.setattr(node_local_files, "close_stores", close_stores)
    monkeypatch.setattr(loop.ray, "get", lambda ref, **kw: ref)
    killed = []
    monkeypatch.setattr(loop.ray, "kill", lambda actor, **kw: killed.append(actor))
    monkeypatch.setattr(loop.ray, "is_initialized", lambda: True)
    shutdown = MagicMock()
    monkeypatch.setattr(loop.ray, "shutdown", shutdown)
    retried = []

    def verify_before_exit(statuses, **kwargs):
        retried.append((
            shutdown.call_count, source not in killed, statuses,
            recorder.record_cleanup.call_args.args[0]["source_teardown"],
        ))
        return {"state": "verified"}

    monkeypatch.setattr(owner_fence, "wait_for_owner_exit", verify_before_exit)
    fence = MagicMock(return_value={"state": "verified"})
    if fence_fails:
        fence.side_effect = TimeoutError("reader still alive")
    monkeypatch.setattr(loop, "wait_for_owner_exit", fence)
    monkeypatch.setattr(loop, "training_loop", MagicMock(side_effect=failure))

    def train(*a, **kw):
        return loop.run_training_loop(
            args, controller, manager, group, [source], cleanup_observer=recorder.record_cleanup,
        )

    monkeypatch.setattr(train_entry, "train_async_no_generation", train)
    with pytest.raises(Exception) as captured:
        train_entry.main()
    cleanup = recorder.record_cleanup.call_args.args[0]
    if fence_fails:
        assert retried == [(0, True, owners, "blocked")]
        shutdown.assert_called_once()
        assert source not in killed
        assert cleanup["source_teardown"] == "blocked"
        assert captured.value.__cause__ is failure
        assert captured.value.owner_statuses == owners
        close_stores.assert_not_called()
    else:
        assert retried == []
        shutdown.assert_called_once()
        assert source in killed
        assert captured.value is failure


@pytest.mark.parametrize("protocol", ["rdma", "nvlink"])
def test_preflight_failure_stops_reader_before_source(monkeypatch, protocol):
    from aspec.ray import cluster_preflight as module

    source, destination, factory = MagicMock(), MagicMock(), MagicMock()
    factory.options.return_value.remote.side_effect = [source, destination]
    monkeypatch.setattr(module.ray, "remote", lambda **kw: lambda cls: factory)
    owner = {"node_id": "2" * 56, "node_ip": "destination", "pid": 123}
    destination.get_owner_status.remote.return_value = owner
    source.stage.remote.return_value = "descriptor"
    destination.read_and_validate.remote.return_value = "blocked-read"
    events = []

    def get(value, **kw):
        if value == "blocked-read":
            raise TimeoutError("native receive stalled")
        return value

    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", lambda handle, **kw: events.append(handle))
    monkeypatch.setattr(
        module, "wait_for_owner_exit",
        lambda statuses, **kw: events.append("fenced"), raising=False,
    )
    args = SimpleNamespace(
        cluster_preflight_timeout_seconds=1,
        inference_node_ips=["source"], training_node_ips=["destination"],
    )
    nodes = {
        "source": {"NodeID": "1" * 56},
        "destination": {"NodeID": "2" * 56},
    }
    with pytest.raises(module.ClusterPreflightError, match="native receive stalled"):
        if protocol == "rdma":
            module._run_pairwise_rdma_probes(
                args, ray_nodes_by_ip=nodes,
                configured_routes={
                    ip: {"device_name": "ib0", "port": 1, "gid_index": 0}
                    for ip in nodes
                },
            )
        else:
            module._run_node_local_nvlink_probes(
                args, ray_nodes_by_ip={"destination": nodes["destination"]},
            )
    assert events == [destination, "fenced", source]


@pytest.mark.parametrize("error", [TimeoutError("still alive"), KeyboardInterrupt("interrupted")])
def test_preflight_failed_fence_retains_source(monkeypatch, error):
    from aspec.ray import cluster_preflight as module

    source, destination = object(), object()
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda actor, **kw: killed.append(actor))
    monkeypatch.setattr(
        module, "wait_for_owner_exit", MagicMock(side_effect=error)
    )
    with pytest.raises(module.SourceTeardownBlockedError) as captured:
        module._terminate_probe_pair(source, destination, {"pid": 123}, timeout=1)
    assert killed == [destination]
    assert captured.value.__cause__ is error


def test_preflight_partial_startup_releases_source(monkeypatch):
    from aspec.ray import cluster_preflight as module

    source = object()
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda actor, **kw: killed.append(actor))
    module._terminate_probe_pair(source, None, None, timeout=1)
    assert killed == [source]


def test_preflight_failed_fence_reaches_main_without_blanket_shutdown(monkeypatch, tmp_path):
    from aspec import train_entry
    from aspec.ray import cluster_preflight, node_local_files, owner_fence

    args = SimpleNamespace(output_dir=str(tmp_path), preflight_only=True)
    source, reader = object(), object()
    owner = {"node_id": "node", "node_ip": "ip", "pid": 123}
    failure = cluster_preflight.ClusterPreflightError("read failed")
    killed = []
    monkeypatch.setattr(train_entry, "_prepare_config", lambda: (args, object()))
    monkeypatch.setattr(train_entry, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(train_entry, "install_termination_signal_handler", lambda: None)
    monkeypatch.setattr(train_entry.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(train_entry.ray, "kill", lambda actor, **kw: killed.append(actor))
    shutdown = MagicMock()
    monkeypatch.setattr(train_entry.ray, "shutdown", shutdown)
    close_stores = MagicMock()
    monkeypatch.setattr(node_local_files, "close_stores", close_stores)
    monkeypatch.setattr(
        cluster_preflight, "wait_for_owner_exit", MagicMock(side_effect=TimeoutError("reader alive")),
    )
    retried = []

    def verify_before_exit(statuses, **kwargs):
        retried.append((shutdown.call_count, source not in killed, statuses))
        return {"state": "verified"}

    monkeypatch.setattr(owner_fence, "wait_for_owner_exit", verify_before_exit)

    def preflight(args):
        try:
            raise failure
        finally:
            cluster_preflight._cleanup_probe_pair(source, reader, owner, timeout=0.1)

    monkeypatch.setattr(train_entry, "run_cluster_preflight", preflight)
    with pytest.raises(cluster_preflight.SourceTeardownBlockedError) as captured:
        train_entry.main()
    assert captured.value.__cause__ is failure
    assert killed == [reader]
    assert retried == [(0, True, [owner])]
    shutdown.assert_called_once()
    close_stores.assert_not_called()
