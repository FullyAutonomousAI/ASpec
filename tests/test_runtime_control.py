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
from unittest import mock

import pytest

from aspec.runtime_control import (
    TerminationRequested,
    _raise_termination,
    is_termination_exception,
    termination_signal_handlers,
)


def test_sigterm_handler_raises_cancellation_exception():
    with pytest.raises(TerminationRequested, match="SIGTERM") as captured:
        _raise_termination(signal.SIGTERM, None)

    assert isinstance(captured.value, KeyboardInterrupt)
    assert captured.value.signum == signal.SIGTERM


def test_ray_sigterm_system_exit_is_recognized_as_cancellation():
    assert is_termination_exception(SystemExit(signal.SIGTERM))
    assert not is_termination_exception(SystemExit(1))


def test_sigterm_handler_is_restored_after_scope():
    previous = signal.getsignal(signal.SIGTERM)
    with termination_signal_handlers():
        assert signal.getsignal(signal.SIGTERM) is _raise_termination
    assert signal.getsignal(signal.SIGTERM) is previous


def test_failed_fence_defers_both_termination_signals_and_restores_handlers():
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in signals}
    with termination_signal_handlers(defer=True):
        for signum in signals:
            assert signal.getsignal(signum) is signal.SIG_IGN
            signal.raise_signal(signum)
    assert {signum: signal.getsignal(signum) for signum in signals} == previous


def test_main_records_sigterm_as_cancelled(monkeypatch, tmp_path):
    from aspec import train_entry

    plan = mock.MagicMock()
    args = mock.MagicMock(output_dir=str(tmp_path))
    recorder = mock.MagicMock()
    monkeypatch.setattr(train_entry, "_prepare_config", lambda: (args, plan))
    monkeypatch.setattr(train_entry, "RunEvidenceRecorder", lambda *_, **__: recorder)
    monkeypatch.setattr(train_entry, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(train_entry.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        train_entry,
        "train_async_no_generation",
        mock.Mock(side_effect=TerminationRequested(signal.SIGTERM)),
    )

    with pytest.raises(TerminationRequested):
        train_entry.main()

    error = recorder.fail.call_args.args[0]
    assert isinstance(error, TerminationRequested)
    recorder.complete.assert_not_called()
