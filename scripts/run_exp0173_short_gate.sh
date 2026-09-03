#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?short-gate result directory required}"
mkdir -p "${result_dir}"

for round in $(seq 1 5); do
    printf -v index '%02d' "${round}"
    if (( round % 2 == 1 )); then
        order=(8 16)
    else
        order=(16 8)
    fi
    for group_tiles in "${order[@]}"; do
        QBH_EXP0173_LM_HEAD_GROUP_TILES="${group_tiles}" \
        QBH_EXP0173_GENERATION_STEPS=17 \
            "${project_root}/scripts/run_exp0173.sh" \
            > "${result_dir}/pair_${index}_batch${group_tiles}.log" 2>&1
    done
done
