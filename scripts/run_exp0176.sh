#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0176 cell required}"

case "${cell}" in
    o_batch4) o_batch_tiles=4 ;;
    o_batch8) o_batch_tiles=8 ;;
    *) printf 'invalid EXP-0176 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac

EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0176-w4u8 \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 \
QBH_EXP0176_O_BATCH_TILES="${o_batch_tiles}" \
    "${project_root}/scripts/run_exp0173.sh"
