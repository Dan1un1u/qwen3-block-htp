#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
schedule="${1:?stage schedule required}"
output_dir="${2:?output directory required}"
rounds="${3:-3}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"

mkdir -p "${output_dir}"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_before.txt"
for variant in F16F16 W4F16; do
    for repeat in 1 10; do
        : > "${output_dir}/${variant,,}_control_r${repeat}.jsonl"
        : > "${output_dir}/${variant,,}_${schedule}_r${repeat}.jsonl"
    done
done

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        variants=(F16F16 W4F16)
        repeats=(1 10)
        schedules=(control "${schedule}")
    else
        variants=(W4F16 F16F16)
        repeats=(10 1)
        schedules=("${schedule}" control)
    fi
    for repeat in "${repeats[@]}"; do
        for variant in "${variants[@]}"; do
            for active_schedule in "${schedules[@]}"; do
                "${project_root}/scripts/run_exp0084.sh" \
                    "${variant}" "${active_schedule}" "${repeat}" on off \
                    >> "${output_dir}/${variant,,}_${active_schedule}_r${repeat}.jsonl"
            done
        done
    done
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_after.txt"
cmp "${output_dir}/boot_id_before.txt" \
    "${output_dir}/boot_id_after.txt"
printf 'OUTPUT_DIR=%s\n' "${output_dir}"
