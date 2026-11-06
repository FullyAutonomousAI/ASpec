#!/usr/bin/env bash
set -euo pipefail
if [[ $# == 0 || ${1:-} == --help ]]; then
  echo 'Usage: bash tools/start_recipe_node.sh RAY_START_ARGUMENTS...'
  echo 'Activate a node-local venv, set RAY_AUTH_TOKEN_PATH to a provisioned private token,'
  echo 'and pass --head --node-ip-address=IP --port=PORT, or --address=HEAD:PORT --node-ip-address=IP.'
  echo 'Runs in the foreground; keep the terminal or supervising job alive.'
  exit 0
fi
if [[ ! -s ${RAY_AUTH_TOKEN_PATH:-} || ! -r ${RAY_AUTH_TOKEN_PATH:-} ]]; then
  echo 'Set RAY_AUTH_TOKEN_PATH to a readable nonempty local token file shared by cluster members.' >&2
  exit 2
fi
export RAY_AUTH_MODE=token
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/tools/run_in_env.sh" ray start --block "$@"
