#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0088_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0088}"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_b_${source_short}"
rounds=5

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before Stage-B collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}"
"${project_root}/scripts/build_exp0088.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/deploy_exp0088_block.sh" > "${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"

for mode in control candidate; do
    "${project_root}/scripts/run_exp0088.sh" "${mode}" 1 on on \
        > "${result_dir}/correctness_${mode}.jsonl"
done
for repeat in 1 10; do
    for mode in control candidate; do
        : > "${result_dir}/paired_${mode}_r${repeat}.jsonl"
    done
done
for ((round = 1; round <= rounds; ++round)); do
    printf 'EXP-0088 Stage-B round %d/%d\n' "${round}" "${rounds}"
    if ((round % 2 == 1)); then
        repeats=(1 10)
        modes=(control candidate)
    else
        repeats=(10 1)
        modes=(candidate control)
    fi
    for repeat in "${repeats[@]}"; do
        for mode in "${modes[@]}"; do
            "${project_root}/scripts/run_exp0088.sh" \
                "${mode}" "${repeat}" on off \
                >> "${result_dir}/paired_${mode}_r${repeat}.jsonl"
        done
    done
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/validate_exp0088_stage_b.py" \
    "${result_dir}" > "${result_dir}/gate_summary.json"
printf 'RESULT_DIR=%s\n' "${result_dir}"
