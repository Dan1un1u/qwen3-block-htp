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
"${project_root}/scripts/check_exp0177_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0177.sh" \
    > "${result_dir}/deploy.log" 2>&1

for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(full row4)
    else
        order=(row4 full)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=17 \
            "${project_root}/scripts/run_exp0177.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done

QBH_EXP0173_GENERATION_STEPS=17 \
    "${project_root}/scripts/run_exp0177.sh" row4_poison \
    > "${result_dir}/raw/padding_poison.log" 2>&1

python3 "${project_root}/scripts/summarize_exp0177_short.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}"
printf 'RESULT_DIR=%s\n' "${result_dir}"
