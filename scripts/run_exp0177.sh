#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:?EXP-0177 cell required}"

case "${cell}" in
    full)
        av_requant_rows=64
        av_padding_poison=0
        ;;
    row4)
        av_requant_rows=4
        av_padding_poison=0
        ;;
    row4_poison)
        av_requant_rows=4
        av_padding_poison=1
        ;;
    *) printf 'invalid EXP-0177 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac

EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0177-w4u8 \
QBH_EXP0173_LM_HEAD_GROUP_TILES=16 \
QBH_EXP0176_O_BATCH_TILES=8 \
QBH_EXP0177_AV_REQUANT_ROWS="${av_requant_rows}" \
QBH_EXP0177_AV_PADDING_POISON="${av_padding_poison}" \
    "${project_root}/scripts/run_exp0173.sh"
