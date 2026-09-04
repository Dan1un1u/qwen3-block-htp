#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?short-gate result directory required}"
source_commit="${2:?source commit required}"

if ! git -C "${project_root}" cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
    printf 'source commit does not exist: %s\n' "${source_commit}" >&2
    exit 2
fi
actual_head="$(git -C "${project_root}" rev-parse HEAD)"
if [[ "${source_commit}" != "${actual_head}" ]]; then
    printf 'source commit mismatch: requested=%s HEAD=%s\n' \
        "${source_commit}" "${actual_head}" >&2
    exit 2
fi

mkdir -p "${result_dir}/raw" \
    "${result_dir}/qk_boundary_control" \
    "${result_dir}/qk_boundary_row4"
"${project_root}/scripts/check_exp0179_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0179.sh" \
    > "${result_dir}/deploy.log" 2>&1

QBH_EXP0173_GENERATION_STEPS=2 \
QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/qk_boundary_control" \
    "${project_root}/scripts/run_exp0179.sh" full \
    > "${result_dir}/qk_boundary_control.log" 2>&1
QBH_EXP0173_GENERATION_STEPS=2 \
QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/qk_boundary_row4" \
    "${project_root}/scripts/run_exp0179.sh" row4 \
    > "${result_dir}/qk_boundary_row4.log" 2>&1

for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(full row4)
    else
        order=(row4 full)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=17 \
            "${project_root}/scripts/run_exp0179.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done

QBH_EXP0173_GENERATION_STEPS=17 \
    "${project_root}/scripts/run_exp0179.sh" row4_poison \
    > "${result_dir}/padding_poison.log" 2>&1

python3 "${project_root}/scripts/summarize_exp0179.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}" \
    --rounds 5 --steps 17
printf 'RESULT_DIR=%s\n' "${result_dir}"
