#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${QBH_EXP0147_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0147}"
capture_root="${QBH_EXP0147_CAPTURE_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0147/golden-captures}"
python_bin="${QBH_EXP0147_PYTHON:-/home/daniuniu/mllm-quant-venv/bin/python}"

mkdir -p "${capture_root}"
cells=(decode_l64 decode_l256 decode_l1024 decode_l4096)
packages=(
    decode_l64_w4u8_v2
    decode_l256_w4u8
    decode_l1024_w4u8
    decode_l4096_w4u8
)
logical_rows=(1 1 1 1)

for index in "${!cells[@]}"; do
    cell="${cells[index]}"
    capture="${capture_root}/${cell}_actual_block_output_u8.bin"
    result="${capture_root}/${cell}_audit_run.json"
    QBH_EXP0147_DEPLOY=1 QBH_DUMP_OUTPUT_PATH="${capture}" \
        "${project_root}/scripts/run_exp0147_w4u8.sh" \
            "${cell}" 1 on off >"${result}"
    "${python_bin}" "${project_root}/scripts/promote_exp0147_device_golden.py" \
        --package "${model_root}/${packages[index]}" \
        --captured-output "${capture}" \
        --logical-rows "${logical_rows[index]}"
done
