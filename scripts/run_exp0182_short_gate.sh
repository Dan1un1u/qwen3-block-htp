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

mkdir -p "${result_dir}/raw"
"${project_root}/scripts/check_exp0182_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0182.sh" \
    > "${result_dir}/deploy.log" 2>&1

for cell in control quartet; do
    QBH_EXP0173_GENERATION_STEPS=34 \
        "${project_root}/scripts/run_exp0182.sh" "${cell}" \
        > "${result_dir}/boundary_${cell}.log" 2>&1
done

for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(control quartet)
    else
        order=(quartet control)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=17 \
            "${project_root}/scripts/run_exp0182.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done

QBH_EXP0173_GENERATION_STEPS=17 \
    "${project_root}/scripts/run_exp0182.sh" quartet_poison \
    > "${result_dir}/padding_poison.log" 2>&1

python3 "${project_root}/scripts/summarize_exp0182.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}" \
    --rounds 5 --steps 17
printf 'RESULT_DIR=%s\n' "${result_dir}"
