#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_EXP0068_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0068}"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/search/${timestamp}_${source_short}"

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}"
"${project_root}/scripts/build_exp0068.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0068_static.sh" > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0068_block.sh" > "${result_dir}/deploy.log" 2>&1
/mnt/c/adb/adb.exe shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
for mode in control context5 context6; do
    : > "${result_dir}/search_${mode}_r10.jsonl"
done
for round in 1 2 3; do
    printf 'EXP-0068 search round %d/3\n' "${round}"
    case "${round}" in
        1) modes=(control context5 context6) ;;
        2) modes=(context6 control context5) ;;
        3) modes=(context5 context6 control) ;;
    esac
    for mode in "${modes[@]}"; do
        "${project_root}/scripts/run_exp0068.sh" "${mode}" 10 on off \
            >> "${result_dir}/search_${mode}_r10.jsonl"
    done
done
/mnt/c/adb/adb.exe shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/summarize_exp0068_search.py" "${result_dir}" \
    > "${result_dir}/selection.json"
printf 'SEARCH_DIR=%s\n' "${result_dir}"
cat "${result_dir}/selection.json"
