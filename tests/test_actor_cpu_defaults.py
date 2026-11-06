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

import pytest

from aspec.utils.env import get_aspec_env_vars

THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def test_actor_cpu_defaults_match_angelspec(monkeypatch):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)

    env = get_aspec_env_vars()

    assert {key: env[key] for key in THREAD_KEYS} == dict.fromkeys(THREAD_KEYS, "8")


@pytest.mark.parametrize("override_key", THREAD_KEYS)
def test_explicit_cpu_environment_overrides_only_its_default(monkeypatch, override_key):
    for key in THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(override_key, "2")

    env = get_aspec_env_vars()

    assert {key: env[key] for key in THREAD_KEYS} == {
        key: "2" if key == override_key else "8" for key in THREAD_KEYS
    }


def test_multinode_preserves_worker_local_library_and_cache_environment(monkeypatch):
    from types import SimpleNamespace

    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "HF_HOME", "CUTE_DSL_CACHE_DIR", "ASPEC_LOG_DIR"):
        monkeypatch.setenv(key, "/driver-only/path")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=configured-route")
    env = get_aspec_env_vars(SimpleNamespace(cluster_mode="static_multi_node"))
    assert env["NCCL_SOCKET_IFNAME"] == "=configured-route"
    assert not any(value == "/driver-only/path" for value in env.values())
    assert get_aspec_env_vars()["LD_LIBRARY_PATH"] == "/driver-only/path"
