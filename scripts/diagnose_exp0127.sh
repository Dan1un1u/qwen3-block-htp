#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_EXP0127_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0127}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_diagnostic"
cells=(control v_batch4 kv_batch4)
mkdir -p "${result_dir}"

for cell in "${cells[@]}"; do
    : > "${result_dir}/${cell}_r1.jsonl"
    : > "${result_dir}/${cell}_r10.jsonl"
done

for round in 1 2 3; do
    if ((round % 2 == 1)); then
        repeats=(1 10)
        order=(control v_batch4 kv_batch4)
    else
        repeats=(10 1)
        order=(kv_batch4 v_batch4 control)
    fi
    for repeat in "${repeats[@]}"; do
        for cell in "${order[@]}"; do
            "${project_root}/scripts/run_exp0127.sh" \
                "${cell}" "${repeat}" on off \
                | grep execution_unit >> "${result_dir}/${cell}_r${repeat}.jsonl"
        done
    done
done

printf 'RESULT_DIR=%s\n' "${result_dir}"
