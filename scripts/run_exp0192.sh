#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0192 cell required}"
case "${cell}" in
    down2) batch_tiles=2 ;;
    down4) batch_tiles=4 ;;
    *) printf 'invalid EXP-0192 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac
EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0192-down-batch4 \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 QBH_EXP0176_O_BATCH_TILES=4 \
QBH_EXP0177_AV_REQUANT_ROWS=4 QBH_EXP0177_AV_PADDING_POISON=0 \
QBH_EXP0178_COMMON_OP_ROWS=4 QBH_EXP0178_COMMON_PADDING_POISON=0 \
QBH_EXP0179_QK_NORM_ROPE_ROWS=4 QBH_EXP0179_QK_PADDING_POISON=0 \
QBH_EXP0183_KV_CACHE_LAYOUT=hmx_native_u8_segmented_vtcm_k7_session_v9 \
QBH_EXP0188_DECODE_PROJECTION_MODE=direct_n \
QBH_W4U8_DECODE_DIRECT_N_MASK=7 \
QBH_EXP0189_SWIGLU_ROWS=4 QBH_EXP0189_SWIGLU_PADDING_POISON=0 \
QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES=16 \
QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES="${batch_tiles}" \
    "${project_root}/scripts/run_exp0173.sh"
