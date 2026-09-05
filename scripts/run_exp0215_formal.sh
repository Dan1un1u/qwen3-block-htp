#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
source_commit="${2:?source commit required}"
audit_result_dir="${3:?short-gate audit result directory required}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
test "$(git -C "${project_root}" rev-parse HEAD)" = "${source_commit}"
test -z "$(git -C "${project_root}" status --porcelain)"
mkdir -p "${result_dir}/raw"
"${project_root}/scripts/build_exp0215.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0215_static.sh" > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0215.sh" > "${result_dir}/deploy.log" 2>&1
for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(control direct_mlp)
    else
        order=(direct_mlp control)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=193 \
        QBH_GENERATION_BOUNDARY_AUDIT=0 \
            "${project_root}/scripts/run_exp0215.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done
"${python_exe}" "${project_root}/scripts/summarize_exp0215.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}" \
    --rounds 10 --steps 193 --audit-result-dir "${audit_result_dir}" \
    --formal --gate
printf 'RESULT_DIR=%s\n' "${result_dir}"
