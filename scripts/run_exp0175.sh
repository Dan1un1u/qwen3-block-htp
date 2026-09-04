#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0175 cell required}"

case "${cell}" in
    post_batch8) gate_up_mode=post_batch8 ;;
    pair_ready_batch8) gate_up_mode=pair_ready_batch8 ;;
    pair_ready_batch16) gate_up_mode=pair_ready_batch16 ;;
    pair_ready2_batch8) gate_up_mode=pair_ready2_batch8 ;;
    pair_ready2_batch16) gate_up_mode=pair_ready2_batch16 ;;
    *) printf 'invalid EXP-0175 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac

EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0175-w4u8 \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 \
QBH_EXP0175_GATE_UP_MODE="${gate_up_mode}" \
    "${project_root}/scripts/run_exp0173.sh"
