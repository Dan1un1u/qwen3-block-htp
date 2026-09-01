#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_EXP0147_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/w4u8}"
mkdir -p "${result_root}"

cells=(
    prefill_m16 prefill_m32 prefill_m64 prefill_m128
    decode_l64 decode_l256 decode_l1024 decode_l4096
)

for cell in "${cells[@]}"; do
    QBH_EXP0147_DEPLOY=1 \
        "${project_root}/scripts/run_exp0147_w4u8.sh" \
            "${cell}" 1 on off >"${result_root}/${cell}_repeat1.json"
    "${project_root}/scripts/run_exp0147_w4u8.sh" \
        "${cell}" 10 on off >"${result_root}/${cell}_repeat10.json"
done

"/home/daniuniu/mllm-quant-venv/bin/python" \
    "${project_root}/scripts/summarize_exp0147.py" \
    --input "${result_root}" \
    --output "${result_root}/summary.json" \
    --markdown "${result_root}/summary.md"
