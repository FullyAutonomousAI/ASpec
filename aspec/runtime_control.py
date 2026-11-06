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

"""Process-level cancellation signals shared by the driver and training loop."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Generator


class TerminationRequested(KeyboardInterrupt):
    """A scheduler requested bounded cancellation of the current run."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = str(signum)
        super().__init__(f"received {self.signal_name}")


def _raise_termination(signum: int, _frame: FrameType | None) -> None:
    raise TerminationRequested(signum)


def install_termination_signal_handler() -> None:
    """Install ASpec's SIGTERM handler after runtimes that replace it initialize."""

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _raise_termination)


def is_termination_exception(error: BaseException | None) -> bool:
    """Recognize both ASpec and Ray representations of scheduler termination."""

    if isinstance(error, TerminationRequested):
        return True
    return isinstance(error, SystemExit) and error.code in {
        signal.SIGTERM,
        -signal.SIGTERM,
    }


@contextmanager
def termination_signal_handlers(*, defer: bool = False) -> Generator[None, None, None]:
    """Handle cancellation, or defer it while a failed reader fence is pending."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    signals = (signal.SIGINT, signal.SIGTERM) if defer else (signal.SIGTERM,)
    previous = {signum: signal.getsignal(signum) for signum in signals}
    if defer:
        # Catching KeyboardInterrupt only around the probe leaves logging and
        # retry bookkeeping interruptible, allowing Ray's unsafe atexit cleanup.
        for signum in signals:
            signal.signal(signum, signal.SIG_IGN)
    else:
        install_termination_signal_handler()
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
