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

"""Prepare this venv's native libraries before executing a fresh process."""
import os
import sys
import sysconfig
from pathlib import Path


def command_environment():
    if sys.platform != "linux" or sys.version_info[:2] < (3, 11):
        raise ValueError("Use Linux with Python >=3.11.")
    active = os.environ.get("VIRTUAL_ENV")
    if not active or sys.prefix == sys.base_prefix or Path(active).resolve() != Path(sys.prefix).resolve():
        raise ValueError("Activate the intended venv; python and VIRTUAL_ENV must agree.")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("Unset PYTHONPATH and PYTHONHOME; use the active isolated venv.")
    settings = dict(line.lower().replace(" ", "").split("=", 1)
                    for line in (Path(sys.prefix) / "pyvenv.cfg").read_text().splitlines()
                    if "=" in line)
    if settings.get("include-system-site-packages") != "false":
        raise ValueError("Use include-system-site-packages=false.")
    env = dict(os.environ)
    paths = []
    if directory := env.get("ASPEC_CUDA_COMPAT_DIR"):
        path = Path(directory).resolve()
        if not (path / "libcuda.so.1").is_file():
            raise ValueError("ASPEC_CUDA_COMPAT_DIR must contain libcuda.so.1.")
        paths.append(str(path))
    purelib = Path(sysconfig.get_paths()["purelib"])
    paths.extend(str(p) for p in (purelib / "nvidia/cuda_runtime/lib", purelib / "torch/lib")
                 if p.is_dir())
    if paths:
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(paths + ([previous] if previous else []))
    if directory := env.get("ASPEC_CACHE_HOME"):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for key, suffix in {
            "XDG_CACHE_HOME": "", "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TORCH_EXTENSIONS_DIR": "torch-extensions",
            "CUDA_CACHE_PATH": "cuda", "CUTE_DSL_CACHE_DIR": "cute",
            "FLASHINFER_WORKSPACE_BASE": "flashinfer", "SGLANG_JIT_CACHE_DIR": "sglang",
        }.items():
            env[key] = str(root / suffix)
    return env


def execute(argv):
    if not argv:
        raise ValueError("Expected a command.")
    os.execvpe(argv[0], argv, command_environment())


if __name__ == "__main__":
    try:
        execute(sys.argv[1:])
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
