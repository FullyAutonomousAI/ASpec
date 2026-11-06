#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

config_path="${1:-configs/recipes/quickstart_qwen3_8b.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$config_path" ]]; then
  printf 'ASpec config does not exist: %s\n' "$config_path" >&2
  exit 2
fi

exec bash tools/run_in_env.sh python -m aspec.train_entry --config "$config_path" "$@"
