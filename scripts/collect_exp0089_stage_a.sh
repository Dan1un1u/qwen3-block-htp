#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0089_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0089}"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_a_${source_short}"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0089 Stage A\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}"
"${project_root}/scripts/build_exp0089.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/deploy_exp0089.sh" > "${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
: > "${result_dir}/timeline.jsonl"
: > "${result_dir}/ordinary.jsonl"
for round in 1 2 3 4 5 6 7; do
    printf 'EXP-0089 Stage-A timeline %d/7\n' "${round}"
    "${project_root}/scripts/run_exp0089.sh" 1 on off 1 \
        >> "${result_dir}/timeline.jsonl"
done
for round in 1 2 3; do
    "${project_root}/scripts/run_exp0089.sh" 1 on off 0 \
        >> "${result_dir}/ordinary.jsonl"
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0089_stage_a.py" \
    "${result_dir}" > "${result_dir}/stage_a_summary.json"
printf 'RESULT_DIR=%s\n' "${result_dir}"
