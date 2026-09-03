#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"

mkdir -p "${result_dir}/raw" "${result_dir}/audit"

QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/audit" \
    "${project_root}/scripts/run_exp0167.sh" \
    > "${result_dir}/audit/device.jsonl" 2>&1

"${project_root}/scripts/verify_exp0167_generation.py" \
    --package-dir "/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/w4u8_greedy16" \
    --audit-dir "${result_dir}/audit" \
    --device-log "${result_dir}/audit/device.jsonl" \
    --output "${result_dir}/audit/independent_reference.json" \
    > "${result_dir}/audit/verify.log" 2>&1

for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    QBH_GENERATION_BOUNDARY_AUDIT=0 \
        "${project_root}/scripts/run_exp0167.sh" \
        > "${result_dir}/raw/generation_${index}.log" 2>&1
done

printf 'RESULT_DIR=%s\n' "${result_dir}"
