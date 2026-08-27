#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0020}"
exec "${project_root}/scripts/run_exp0019.sh" "$@"
