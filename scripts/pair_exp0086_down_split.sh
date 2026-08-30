#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?result directory required}"
rounds="${2:-5}"
mkdir -p "${result_dir}"

for candidate in 64 80 112 128; do
    for repeat in 1 10; do
        : > "${result_dir}/candidate${candidate}_control_repeat${repeat}.jsonl"
        : > "${result_dir}/candidate${candidate}_candidate_repeat${repeat}.jsonl"
    done
    for ((round = 1; round <= rounds; ++round)); do
        if ((round % 2 == 1)); then
            order=(96 "${candidate}")
        else
            order=("${candidate}" 96)
        fi
        for split in "${order[@]}"; do
            if [[ "${split}" == 96 ]]; then
                role=control
            else
                role=candidate
            fi
            "${project_root}/scripts/run_exp0086.sh" \
                "${split}" 1 on off \
                >> "${result_dir}/candidate${candidate}_${role}_repeat1.jsonl"
            "${project_root}/scripts/run_exp0086.sh" \
                "${split}" 10 on off \
                >> "${result_dir}/candidate${candidate}_${role}_repeat10.jsonl"
        done
    done
done
