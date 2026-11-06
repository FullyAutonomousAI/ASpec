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

"""Opt-in process-exit check in a private CPU-only Ray instance."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import psutil
import pytest


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_RAY_INTEGRATION") != "1",
    reason="set ASPEC_RUN_RAY_INTEGRATION=1 to start a private CPU-only Ray instance",
)
def test_cli_retains_real_source_actor_until_reader_process_exits(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""\
        import atexit
        import json
        import os
        import sys
        import threading
        import time
        from pathlib import Path

        import ray
        from aspec.ray import owner_fence
        from aspec.ray.owner_fence import SourceTeardownBlockedError, run_with_source_fencing

        root = Path(sys.argv[1])
        atexit.register(lambda: (root / "exited").touch())
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
                 _temp_dir=str(root / "ray"))

        @ray.remote(num_cpus=0)
        class Owner:
            def __init__(self):
                self.buffer = bytearray(4096)

            def status(self):
                import aspec
                return {"pid": os.getpid(), "node_id": ray.get_runtime_context().get_node_id(),
                        "node_ip": ray.util.get_node_ip_address(), "module": aspec.__file__,
                        "executable": sys.executable}

            def read(self):
                (root / "reading").touch()
                while True:
                    time.sleep(0.05)

        source, reader = Owner.remote(), Owner.remote()
        statuses = ray.get([source.status.remote(), reader.status.remote()], timeout=60)
        reading = reader.read.remote()
        while not (root / "reading").exists():
            time.sleep(0.05)
        (root / "owners.json").write_text(json.dumps(statuses))

        def external_fence():
            while not (root / "release").exists():
                time.sleep(0.05)
            ray.kill(reader, no_restart=True)

        threading.Thread(target=external_fence, daemon=True).start()

        original_wait = owner_fence.wait_for_owner_exit
        def observed_wait(*args, **kwargs):
            (root / "guarded").touch()
            return original_wait(*args, **kwargs)
        owner_fence.wait_for_owner_exit = observed_wait

        def failed_entry():
            try:
                raise ValueError("original training failure")
            except ValueError as failure:
                raise SourceTeardownBlockedError([statuses[1]]) from failure

        run_with_source_fencing(failed_entry)
        """)
    # Short paths also keep Ray's Unix-domain sockets below the OS limit.
    with tempfile.TemporaryDirectory(prefix="aspec-fence-") as temporary:
        root = Path(temporary)
        log_path = tmp_path / "driver.log"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("RAY_ADDRESS", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(root)], cwd=repo, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 90
                while not (root / "guarded").exists():
                    assert process.poll() is None, log_path.read_text()
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.05)
                statuses = json.loads((root / "owners.json").read_text())
                for status in statuses:
                    assert Path(status["module"]).resolve().is_relative_to(repo)
                    assert Path(status["executable"]).resolve() == Path(sys.executable).resolve()
                time.sleep(0.5)
                os.kill(process.pid, signal.SIGINT)
                os.kill(process.pid, signal.SIGTERM)
                time.sleep(0.2)
                assert process.poll() is None, log_path.read_text()
                assert not (root / "exited").exists()
                assert all(psutil.pid_exists(status["pid"]) for status in statuses)
                (root / "release").touch()
                assert process.wait(timeout=60) != 0
                assert (root / "exited").exists()
                output = log_path.read_text()
                assert "SourceTeardownBlockedError" in output
                assert "original training failure" in output
                deadline = time.monotonic() + 10
                while any(psutil.pid_exists(status["pid"]) for status in statuses):
                    assert time.monotonic() < deadline, "private Ray owners did not exit"
                    time.sleep(0.05)
            finally:
                (root / "release").touch()
                if process.poll() is None:
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        # Only this test's private session; never a shared Ray job.
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
