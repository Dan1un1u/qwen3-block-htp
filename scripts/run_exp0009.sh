#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0009}"

storage="${1:-packed_w4_hmx_postscale}"
projection="${2:-down}"
pattern="${3:-identity}"
repeat_count="${4:-1}"
physical_plan="${5:-slots4_chunk96}"

exec "${project_root}/scripts/run_exp0006.sh" \
    "${storage}" "${projection}" "${pattern}" \
    "${repeat_count}" "${physical_plan}"
