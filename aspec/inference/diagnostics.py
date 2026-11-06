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

"""Persist backend child-process output independently of Ray actor survival."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def capture_backend_output(path):
    """Child processes inherit these descriptors even if their actor exits."""
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with path.open("ab", buffering=0) as stream:
            for fd in (1, 2):
                os.dup2(stream.fileno(), fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def backend_error_logs(output_dir):
    """Bound failure Evidence while retaining paths to full backend logs."""
    records = []
    for path in sorted((Path(output_dir) / "backend_logs").glob("sglang-*.log"))[:16]:
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 8192))
                tail = stream.read(8192).decode("utf-8", errors="replace")
            records.append({"path": str(path), "tail": tail, "truncated": size > 8192})
        except OSError as exc:
            records.append({"path": str(path), "read_error": str(exc)})
    return records
