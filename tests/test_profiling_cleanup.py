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
from unittest.mock import Mock

import pytest

from aspec.utils import profiling


@pytest.mark.parametrize("mode", ["complete", "close", "error"])
def test_profile_iterator_stops(monkeypatch, mode):
    profiler = Mock()
    monkeypatch.setattr(profiling, "_create_torch_profiler", lambda *a, **k: profiler)

    def items():
        yield 1
        if mode == "error":
            raise ValueError("iterator failed")
        yield 2

    gen = profiling._profile_simple_loop(
        items(),
        SimpleNamespace(use_pytorch_profiler=True, profile_target=["train_actor"]),
        "train_actor",
    )
    if mode == "close":
        next(gen)
        gen.close()
    elif mode == "error":
        with pytest.raises(ValueError, match="iterator failed"):
            list(gen)
    else:
        assert list(gen) == [1, 2]
    profiler.stop.assert_called_once()


@pytest.mark.parametrize("early_stop", [False, True])
def test_overall_close_is_idempotent_and_stops_memory_once(monkeypatch, early_stop):
    torch_profiler, memory = Mock(), Mock()
    monkeypatch.setattr(profiling, "_create_torch_profiler", lambda *a, **k: torch_profiler)
    monkeypatch.setattr(profiling._BaseMemoryProfiler, "create", lambda args: memory)
    args = SimpleNamespace(
        use_pytorch_profiler=True,
        record_memory_history=True,
        profile_target=["train_overall"],
        memory_snapshot_num_steps=2,
    )
    profiler = profiling.TrainProfiler(args)
    profiler.on_init_end()
    profiler.on_init_end()
    if not early_stop:
        profiler.step(1)
    profiler.close()
    profiler.close()
    torch_profiler.start.assert_called_once()
    torch_profiler.stop.assert_called_once()
    memory.start.assert_called_once()
    memory.stop.assert_called_once()


def test_memory_history_disables_even_when_snapshot_fails(monkeypatch):
    profiler = profiling._TorchMemoryProfiler.__new__(profiling._TorchMemoryProfiler)
    profiler._path_dump = "unused"
    monkeypatch.setattr(
        profiling.torch.cuda.memory, "_dump_snapshot", Mock(side_effect=OSError("disk full"))
    )
    recorder = Mock()
    monkeypatch.setattr(profiling.torch.cuda.memory, "_record_memory_history", recorder)
    with pytest.raises(OSError, match="disk full"):
        profiler.stop()
    recorder.assert_called_once_with(enabled=None)
