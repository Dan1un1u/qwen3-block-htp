#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="${1:?f16f16 or w4f16 required}"
result_root="${QBH_EXP0147_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/${recipe}}"
mkdir -p "${result_root}"

case "${recipe}" in
    f16f16|w4f16) ;;
    *) printf 'unknown recipe: %s\n' "${recipe}" >&2; exit 2 ;;
esac

cells=(
    prefill_m16 prefill_m32 prefill_m64 prefill_m128
    decode_l64 decode_l256 decode_l1024 decode_l4096
)

for cell in "${cells[@]}"; do
    QBH_EXP0147_DEPLOY=1 \
        "${project_root}/scripts/run_exp0147_fp16.sh" \
            "${recipe}" "${cell}" 1 on off \
            >"${result_root}/${cell}_repeat1.json"
    "${project_root}/scripts/run_exp0147_fp16.sh" \
        "${recipe}" "${cell}" 10 on off \
        >"${result_root}/${cell}_repeat10.json"
done

"/home/daniuniu/mllm-quant-venv/bin/python" \
    "${project_root}/scripts/summarize_exp0147.py" \
    --input "${result_root}" \
    --output "${result_root}/summary.json" \
    --markdown "${result_root}/summary.md" \
    --label "${recipe}"
