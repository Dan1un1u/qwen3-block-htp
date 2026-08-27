#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0011}"

storage="${1:-packed_w4_hmx_postscale}"
projection="${2:-gate_up}"
pattern="${3:-identity}"
repeat_count="${4:-1}"
physical_plan="${5:-slots4_chunk64_dma_batch4}"

exec "${project_root}/scripts/poll_exp0006.sh" \
    "${storage}" "${projection}" "${pattern}" \
    "${repeat_count}" "${physical_plan}"
