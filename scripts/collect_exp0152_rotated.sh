#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${1:?EXP-0152 model root required}"
w4u8_package="${2:?exact-reference W4U8 package required}"
output="${3:?output directory required}"
rounds="${4:-10}"

recipes=(f16f16 w4f16 w4u8)
mkdir -p "${output}"
for ((round = 1; round <= rounds; ++round)); do
    printf -v label '%02d' "${round}"
    rotation=$(((round - 1) % 3))
    for ((slot = 0; slot < 3; ++slot)); do
        recipe="${recipes[$(((slot + rotation) % 3))]}"
        case "${recipe}" in
            f16f16|w4f16)
                package="${model_root}/${recipe}"
                remote_root="/data/local/tmp/qwen3-block-htp/exp0152-${recipe}"
                ;;
            w4u8)
                package="${w4u8_package}"
                remote_root="/data/local/tmp/qwen3-block-htp/exp0152-w4u8-exact-ref-v1"
                ;;
        esac
        EXP0152_REMOTE_ROOT="${remote_root}" \
        QBH_EXP0152_PACKAGE="${package}" \
            "${project_root}/scripts/run_exp0152.sh" \
            "${recipe}" replay \
            > "${output}/r${label}_${recipe}.log" 2>&1
        printf 'completed_%s_%s\n' "${label}" "${recipe}"
    done
done
