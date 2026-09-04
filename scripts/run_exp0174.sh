#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:-post_batch8}"

case "${cell}" in
post_batch8|inline_batch8|inline_batch16) ;;
*) printf 'usage: %s post_batch8|inline_batch8|inline_batch16\n' "$0" >&2; exit 2 ;;
esac

export EXP0173_REMOTE_ROOT="${EXP0174_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0173-exp0174-w4u8}"
export QBH_EXP0173_LM_HEAD_GROUP_TILES=16
export QBH_W4U8_DECODE_GATE_UP="${cell}"
export QBH_BLOCK_EXPERIMENT=174
exec "${project_root}/scripts/run_exp0173.sh"
