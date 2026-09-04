#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0188 cell required}"
case "${cell}" in
    expanded_s8) projection_mode=expanded_s8; direct_n_mask=0 ;;
    direct_n)
        projection_mode=direct_n
        # QKV, O and MLP use direct-n.  LM-head remains on the accepted
        # group16 expanded-S8 path: the group4 direct-n diagnostic changed
        # U8 argmax tie-breaking and was slower end to end.
        direct_n_mask="${QBH_EXP0188_DIRECT_N_MASK:-7}"
        ;;
    *) printf 'invalid EXP-0188 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac
EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0188-direct-n \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 QBH_EXP0176_O_BATCH_TILES=4 \
QBH_EXP0177_AV_REQUANT_ROWS=4 QBH_EXP0177_AV_PADDING_POISON=0 \
QBH_EXP0178_COMMON_OP_ROWS=4 QBH_EXP0178_COMMON_PADDING_POISON=0 \
QBH_EXP0179_QK_NORM_ROPE_ROWS=4 QBH_EXP0179_QK_PADDING_POISON=0 \
QBH_EXP0183_KV_CACHE_LAYOUT=hmx_native_u8_segmented_vtcm_k7_session_v9 \
QBH_EXP0188_DECODE_PROJECTION_MODE="${projection_mode}" \
QBH_W4U8_DECODE_DIRECT_N_MASK="${direct_n_mask}" \
    "${project_root}/scripts/run_exp0173.sh"
