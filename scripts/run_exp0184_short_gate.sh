#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?short-gate result directory required}"
source_commit="${2:?source commit required}"
test "$(git -C "${project_root}" rev-parse HEAD)" = "${source_commit}"
mkdir -p "${result_dir}/raw"
"${project_root}/scripts/check_exp0184_static.sh" > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0184.sh" > "${result_dir}/deploy.log" 2>&1
for cell in control k_tail; do
    out_cell="$([[ "${cell}" == k_tail ]] && printf quartet || printf control)"
    QBH_EXP0173_GENERATION_STEPS=34 "${project_root}/scripts/run_exp0184.sh" "${cell}" > "${result_dir}/boundary_${out_cell}.log" 2>&1
done
for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then order=(control k_tail); else order=(k_tail control); fi
    for cell in "${order[@]}"; do
        out_cell="$([[ "${cell}" == k_tail ]] && printf quartet || printf control)"
        QBH_EXP0173_GENERATION_STEPS=17 "${project_root}/scripts/run_exp0184.sh" "${cell}" > "${result_dir}/raw/pair_${index}_${out_cell}.log" 2>&1
    done
done
QBH_EXP0173_GENERATION_STEPS=17 "${project_root}/scripts/run_exp0184.sh" k_tail_poison > "${result_dir}/padding_poison.log" 2>&1
python3 "${project_root}/scripts/summarize_exp0184.py" --result-dir "${result_dir}" --source-commit "${source_commit}" --rounds 5 --steps 17
printf 'RESULT_DIR=%s\n' "${result_dir}"
