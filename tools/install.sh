#!/usr/bin/env bash
set -euo pipefail
dev=0
gpu=()
for argument in "$@"; do
  case "$argument" in
    --dev) dev=1 ;;
    --gpu) gpu=(--gpu) ;;
    --help)
      echo 'Usage: bash tools/install.sh [--dev] [--gpu]'
      echo 'For an already activated isolated venv. New users: bash setup.sh'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
bash tools/run_in_env.sh true
command -v git >/dev/null || { echo 'Install Git before setup.' >&2; exit 2; }
python - <<'CHECK'
import ctypes
import os
import platform
import shutil
from pathlib import Path
name, version = platform.libc_ver()
if name != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 34):
    raise SystemExit('Pinned SGLang wheels require glibc >= 2.34; use a newer Linux base image.')
if platform.machine() != 'x86_64':
    raise SystemExit('The pinned runtime requires Linux x86-64.')
cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
nvcc = str(Path(cuda_root) / "bin/nvcc") if cuda_root else shutil.which("nvcc")
if not nvcc or not os.access(nvcc, os.X_OK):
    raise SystemExit("SGLang JIT requires a CUDA development toolkit with nvcc. "
                     "Set CUDA_HOME to that toolkit or use the development container.")
for name in ('libibverbs.so.1', 'libmlx5.so.1'):
    try:
        ctypes.CDLL(name)
    except OSError as error:
        raise SystemExit(f'{name}: {error}. Install rdma-core runtime libraries '
                         '(Ubuntu: libibverbs1 libmlx5-1), or use the container.') from error
CHECK
if ! printf '#include <bit>\nint main() { return std::bit_cast<unsigned>(1.0f) == 0; }\n' |
  "${CXX:-c++}" -std=c++20 -x c++ -fsyntax-only -; then
  echo 'SGLang JIT needs a C++20 compiler and matching standard library.' >&2
  echo 'Install a modern C++ toolchain (Ubuntu: g++), or set CXX to one; see installation-advanced.md.' >&2
  exit 2
fi
extras=fa
(( dev == 0 )) || extras=dev,$extras
python -m pip install uv==0.13.5
python -m uv pip install --link-mode copy --python "$VIRTUAL_ENV/bin/python" \
  --override requirements/cute-overrides.txt -r requirements/sglang-v0.5.21.txt -e ".[${extras}]"
aspec patch-sglang
aspec patch-sglang --check
bash tools/run_in_env.sh python -m tools.doctor "${gpu[@]}"
