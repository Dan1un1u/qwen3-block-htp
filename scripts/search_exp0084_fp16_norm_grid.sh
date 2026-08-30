#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?output directory required}"
rounds="${2:-2}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
rows_grid=(2 4 8)
contexts_grid=(2 3 4)

mkdir -p "${output_dir}"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_before.txt"
for variant in F16F16 W4F16; do
    for repeat in 1 10; do
        for rows in "${rows_grid[@]}"; do
            for contexts in "${contexts_grid[@]}"; do
                : > "${output_dir}/${variant,,}_r${rows}_c${contexts}_repeat${repeat}.jsonl"
            done
        done
    done
done

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        variants=(F16F16 W4F16)
        repeats=(1 10)
        rows_order=(2 4 8)
        contexts_order=(2 3 4)
    else
        variants=(W4F16 F16F16)
        repeats=(10 1)
        rows_order=(8 4 2)
        contexts_order=(4 3 2)
    fi
    for repeat in "${repeats[@]}"; do
        for rows in "${rows_order[@]}"; do
            for contexts in "${contexts_order[@]}"; do
                for variant in "${variants[@]}"; do
                    EXP0084_FP16_NORM_ROWS="${rows}" \
                    EXP0084_FP16_NORM_CONTEXTS="${contexts}" \
                        "${project_root}/scripts/run_exp0084.sh" \
                            "${variant}" candidate "${repeat}" on off \
                            >> "${output_dir}/${variant,,}_r${rows}_c${contexts}_repeat${repeat}.jsonl"
                done
            done
        done
    done
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_after.txt"
cmp "${output_dir}/boot_id_before.txt" \
    "${output_dir}/boot_id_after.txt"
printf 'OUTPUT_DIR=%s\n' "${output_dir}"
