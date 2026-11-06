#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash tools/run_in_env.sh COMMAND [ARG ...]'
  echo 'Activate an isolated ASpec venv first. Preserves arguments and working directory.'
  echo 'Optional ASPEC_CUDA_COMPAT_DIR and ASPEC_CACHE_HOME configure private libraries/caches.'
  exit 0
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$repo_root/aspec/_environment.py" "$@"
