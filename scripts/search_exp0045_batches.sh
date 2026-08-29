#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:-/tmp/exp0045-batch-search}"
rounds="${2:-5}"
modes=(serial qkv_batch2 qkv_batch4)
repeats=(1 10)

mkdir -p "${result_dir}"
for repeat in "${repeats[@]}"; do
    for mode in "${modes[@]}"; do
        : > "${result_dir}/${mode}_r${repeat}.jsonl"
    done
done

for round in $(seq 1 "${rounds}"); do
    if (( round % 2 == 1 )); then
        repeat_order=(1 10)
        mode_order=(serial qkv_batch2 qkv_batch4)
    else
        repeat_order=(10 1)
        mode_order=(qkv_batch4 qkv_batch2 serial)
    fi
    for repeat in "${repeat_order[@]}"; do
        for mode in "${mode_order[@]}"; do
            "${project_root}/scripts/run_exp0045_block.sh" \
                "${mode}" "${repeat}" on off \
                >> "${result_dir}/${mode}_r${repeat}.jsonl"
        done
    done
done

python3 "${project_root}/scripts/summarize_exp0045_search.py" \
    "${result_dir}" | tee "${result_dir}/summary.json"
