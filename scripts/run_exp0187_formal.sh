#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
source_commit="${2:-WORKTREE}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"

if [[ "${source_commit}" != WORKTREE ]]; then
    test "$(git -C "${project_root}" rev-parse HEAD)" = "${source_commit}"
fi
mkdir -p "${result_dir}/raw"
"${project_root}/scripts/check_exp0187_static.sh" > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0187.sh" > "${result_dir}/deploy.log" 2>&1

for repeat_count in 1 10; do
    for projection in gate_up_pair down; do
        for round in $(seq 1 10); do
            printf -v index '%02d' "${round}"
            if ((round % 2 == 1)); then
                order=(control direct_n)
            else
                order=(direct_n control)
            fi
            for cell in "${order[@]}"; do
                "${project_root}/scripts/run_exp0187.sh" \
                    "${cell}" "${projection}" "${repeat_count}" \
                    > "${result_dir}/raw/${projection}_r${repeat_count}_${index}_${cell}.log" 2>&1
            done
        done
    done
done

"${python_exe}" "${project_root}/scripts/summarize_exp0187.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}"
printf 'RESULT_DIR=%s\n' "${result_dir}"
