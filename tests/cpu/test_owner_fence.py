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

import signal
from unittest.mock import MagicMock

import pytest

from aspec.ray import owner_fence
from aspec.runtime_control import TerminationRequested


def test_lost_domain_is_explicit_and_requires_node_local_fail_stop(monkeypatch):
    monkeypatch.setattr(
        owner_fence.ray,
        "nodes",
        lambda: [{"NodeID": "lost", "NodeManagerAddress": "192.0.2.2", "Alive": False}],
    )
    statuses = [{"node_id": "lost", "node_ip": "192.0.2.2", "pid": 123}]
    with pytest.raises(RuntimeError, match="Cannot prove"):
        owner_fence.wait_for_owner_exit(statuses)
    proof = owner_fence.wait_for_owner_exit(statuses, allow_node_loss=True)
    assert proof["lost_node_ids"] == ["lost"]
    assert proof["physical_memory_unobserved_nodes"] == ["lost"]


def test_alive_domain_requires_real_pid_exit_observation(monkeypatch):
    monkeypatch.setattr(
        owner_fence.ray,
        "nodes",
        lambda: [{"NodeID": "a" * 56, "NodeManagerAddress": "192.0.2.1", "Alive": True}],
    )
    probe = MagicMock()
    probe.exited.remote.return_value = True
    remote = MagicMock()
    remote.options.return_value.remote.return_value = probe
    monkeypatch.setattr(owner_fence, "_OwnerFenceProbe", remote)
    monkeypatch.setattr(owner_fence.ray, "get", lambda ref, **kw: ref)
    monkeypatch.setattr(owner_fence.ray, "kill", MagicMock())
    proof = owner_fence.wait_for_owner_exit(
        [{"node_id": "a" * 56, "node_ip": "192.0.2.1", "pid": 123}]
    )
    assert proof["live_node_processes"] == 1
    probe.exited.remote.assert_called_with([123])


@pytest.mark.parametrize("statuses", [None, [], [{"pid": 12}]])
def test_missing_owner_identity_cannot_clear_ownership(statuses):
    with pytest.raises(ValueError, match="identit"):
        owner_fence.wait_for_owner_exit(statuses)


def test_node_death_can_follow_an_unavailable_process_probe(monkeypatch):
    nodes = iter(
        [
            [{"NodeID": "b" * 56, "NodeManagerAddress": "192.0.2.2", "Alive": True}],
            [{"NodeID": "b" * 56, "NodeManagerAddress": "192.0.2.2", "Alive": False}],
        ]
    )
    monkeypatch.setattr(owner_fence.ray, "nodes", lambda: next(nodes))
    probe = MagicMock()
    remote = MagicMock()
    remote.options.return_value.remote.return_value = probe
    monkeypatch.setattr(owner_fence, "_OwnerFenceProbe", remote)

    def unavailable(*args, **kwargs):
        raise owner_fence.ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(owner_fence.ray, "get", unavailable)
    monkeypatch.setattr(owner_fence.ray, "kill", MagicMock())
    monkeypatch.setattr(owner_fence.time, "sleep", lambda _: None)
    proof = owner_fence.wait_for_owner_exit(
        [{"node_id": "b" * 56, "node_ip": "192.0.2.2", "pid": 123}], allow_node_loss=True
    )
    assert proof["lost_node_ids"] == ["b" * 56]
    assert proof["live_node_processes"] == 0


def test_cli_waits_for_readers_before_implicit_ray_teardown(monkeypatch):
    owners = [{"node_id": "node", "node_ip": "ip", "pid": 123}]
    blocked = owner_fence.SourceTeardownBlockedError(owners, allow_node_loss=True)
    events = []
    calls = []
    results = iter([TimeoutError("reader alive"), TerminationRequested(signal.SIGTERM), {}])

    def fence(statuses, **kwargs):
        calls.append((statuses, kwargs))
        result = next(results, None)
        events.append("wait")
        if isinstance(result, BaseException):
            raise result
        events.append("fenced")
        return result

    def fail():
        raise blocked

    monkeypatch.setattr(owner_fence, "wait_for_owner_exit", fence)
    monkeypatch.setattr(owner_fence.time, "sleep", lambda _: None)
    monkeypatch.setattr(owner_fence.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(owner_fence.ray, "shutdown", lambda: events.append("shutdown"))
    with pytest.raises(owner_fence.SourceTeardownBlockedError) as captured:
        owner_fence.run_with_source_fencing(fail)
    assert captured.value is blocked
    assert calls == [(owners, {"allow_node_loss": True})] * 3
    assert events == ["wait", "wait", "wait", "fenced", "shutdown"]


@pytest.mark.parametrize("failure", [None, ValueError("ordinary failure")])
def test_cli_guard_preserves_normal_entry_behavior(monkeypatch, failure):
    fence = MagicMock()
    shutdown = MagicMock()
    monkeypatch.setattr(owner_fence, "wait_for_owner_exit", fence)
    monkeypatch.setattr(owner_fence.ray, "shutdown", shutdown)

    def entry():
        if failure is not None:
            raise failure
        return 42

    if failure is None:
        assert owner_fence.run_with_source_fencing(entry) == 42
    else:
        with pytest.raises(ValueError) as captured:
            owner_fence.run_with_source_fencing(entry)
        assert captured.value is failure
    fence.assert_not_called()
    shutdown.assert_not_called()
