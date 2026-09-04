#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0182 cell required}"

case "${cell}" in
    control)
        cache_layout=hmx_native_u8_segmented_v4
        qk_padding_poison=0
        ;;
    quartet|vtcm_tail)
        cache_layout=hmx_native_u8_segmented_vtcm_tail_v7
        qk_padding_poison=0
        ;;
    quartet_poison|vtcm_tail_poison)
        cache_layout=hmx_native_u8_segmented_vtcm_tail_v7
        qk_padding_poison=1
        ;;
    *) printf 'invalid EXP-0182 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac

EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0182-w4u8 \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 \
QBH_EXP0176_O_BATCH_TILES=8 \
QBH_EXP0177_AV_REQUANT_ROWS=4 \
QBH_EXP0177_AV_PADDING_POISON=0 \
QBH_EXP0178_COMMON_OP_ROWS=4 \
QBH_EXP0178_COMMON_PADDING_POISON=0 \
QBH_EXP0179_QK_NORM_ROPE_ROWS=4 \
QBH_EXP0179_QK_PADDING_POISON="${qk_padding_poison}" \
QBH_EXP0182_KV_CACHE_LAYOUT="${cache_layout}" \
    "${project_root}/scripts/run_exp0173.sh"
