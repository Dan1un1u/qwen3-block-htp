#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?result directory required}"
rounds="${2:-3}"
mkdir -p "${result_dir}"

for split in 64 80 96 112 128; do
    : > "${result_dir}/split${split}_repeat1.jsonl"
    : > "${result_dir}/split${split}_repeat10.jsonl"
done

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        order=(64 80 96 112 128)
    else
        order=(128 112 96 80 64)
    fi
    for split in "${order[@]}"; do
        "${project_root}/scripts/run_exp0086.sh" \
            "${split}" 1 on off \
            >> "${result_dir}/split${split}_repeat1.jsonl"
        "${project_root}/scripts/run_exp0086.sh" \
            "${split}" 10 on off \
            >> "${result_dir}/split${split}_repeat10.jsonl"
    done
done
