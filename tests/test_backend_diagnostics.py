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

import os
import subprocess
import sys

import pytest

from aspec.inference.diagnostics import backend_error_logs, capture_backend_output


def test_child_failure_is_preserved_and_parent_descriptors_are_restored(tmp_path, capfd):
    log = tmp_path / "backend_logs" / "sglang-0-0-0.log"
    with pytest.raises(RuntimeError):
        with capture_backend_output(log):
            subprocess.run([sys.executable, "-c", "import os; os.write(2, b'EADDRINUSE port: 47966\\n')"], check=True)
            raise RuntimeError("actor unavailable")
    os.write(2, b"parent restored\n")
    assert "parent restored" in capfd.readouterr().err
    assert "parent restored" not in log.read_text()
    result = backend_error_logs(tmp_path)
    assert len(result) == 1
    assert "EADDRINUSE port: 47966" in result[0]["tail"]
    assert result[0]["truncated"] is False


def test_backend_log_read_is_bounded_and_missing_directory_is_allowed(tmp_path):
    assert backend_error_logs(tmp_path) == []
    directory = tmp_path / "backend_logs"
    directory.mkdir()
    for i in range(20):
        (directory / f"sglang-{i}.log").write_text("x" * 10000 + "EADDRINUSE")
    logs = backend_error_logs(tmp_path)
    assert len(logs) == 16
    assert all(len(item["tail"].encode()) == 8192 and item["truncated"] for item in logs)
