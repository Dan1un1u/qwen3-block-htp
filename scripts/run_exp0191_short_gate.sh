#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?short-gate result directory required}"
source_commit="${2:?source commit required}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"

test "$(git -C "${project_root}" rev-parse HEAD)" = "${source_commit}"
test -z "$(git -C "${project_root}" status --porcelain)"
mkdir -p "${result_dir}/raw" "${result_dir}/audit"

"${project_root}/scripts/build_exp0191.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0191_static.sh" > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0191.sh" > "${result_dir}/deploy.log" 2>&1

for cell in batch8 batch16; do
    QBH_EXP0173_GENERATION_STEPS=4 \
    QBH_GENERATION_BOUNDARY_AUDIT=1 \
    QBH_GENERATION_AUDIT_DIR="${result_dir}/audit/${cell}" \
        "${project_root}/scripts/run_exp0191.sh" "${cell}" \
        > "${result_dir}/raw/audit_${cell}.log" 2>&1
done

for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(batch8 batch16)
    else
        order=(batch16 batch8)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=193 \
        QBH_GENERATION_BOUNDARY_AUDIT=0 \
            "${project_root}/scripts/run_exp0191.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done

"${python_exe}" "${project_root}/scripts/summarize_exp0190.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}" \
    --rounds 5 --steps 193 --require-audit --gate \
    --experiment 191 --control-batch 8 --candidate-batch 16
printf 'RESULT_DIR=%s\n' "${result_dir}"
