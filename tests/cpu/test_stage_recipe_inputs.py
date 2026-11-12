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
from pathlib import Path

from tools.stage_recipe_inputs import checksum_lines


def test_stage_and_verify_with_local_ssh_double(tmp_path):
    # Execute the real tar/sha256sum pipeline in isolated directories; no network.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text('#!/usr/bin/env bash\nset -e\ncommand="${@: -1}"\nexec sh -c "$command"\n')
    ssh.chmod(0o755)
    source = tmp_path / "source model"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    (source / "config.json").write_text("{}")
    destination = tmp_path / "remote node" / "model;quoted"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    cmd = [
        sys.executable,
        "-m",
        "tools.stage_recipe_inputs",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--host",
        "fake-node",
    ]
    root = Path(__file__).resolve().parents[2]
    subprocess.run(cmd, env=env, cwd=root, check=True, capture_output=True)
    assert checksum_lines(source) == checksum_lines(destination)
    assert subprocess.run(cmd, env=env, cwd=root, capture_output=True).returncode != 0
    subprocess.run(cmd + ["--verify-only"], env=env, cwd=root, check=True, capture_output=True)
    (destination / "weights.bin").write_bytes(b"changed")
    assert (
        subprocess.run(cmd + ["--verify-only"], env=env, cwd=root, capture_output=True).returncode
        != 0
    )
