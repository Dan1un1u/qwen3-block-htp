#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?output directory required}"
common_rows="${2:?common rows/task required}"
common_contexts="${3:?common contexts required}"
f16_best_rows="${4:?F16F16 best rows/task required}"
f16_best_contexts="${5:?F16F16 best contexts required}"
w4_best_rows="${6:?W4F16 best rows/task required}"
w4_best_contexts="${7:?W4F16 best contexts required}"
rounds="${8:-7}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"

mkdir -p "${output_dir}/canonical" "${output_dir}/recipe_best"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_before.txt"
for plan in canonical recipe_best; do
    for variant in f16f16 w4f16 w4u8; do
        for repeat in 1 10; do
            : > "${output_dir}/${plan}/${variant}_r${repeat}.jsonl"
        done
    done
done

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        plans=(canonical recipe_best)
        repeats=(1 10)
    else
        plans=(recipe_best canonical)
        repeats=(10 1)
    fi
    case $(((round - 1) % 3)) in
        0) variants=(F16F16 W4F16 W4U8) ;;
        1) variants=(W4F16 W4U8 F16F16) ;;
        2) variants=(W4U8 F16F16 W4F16) ;;
    esac
    for plan in "${plans[@]}"; do
        for repeat in "${repeats[@]}"; do
            for variant in "${variants[@]}"; do
                rows="${common_rows}"
                contexts="${common_contexts}"
                if [[ "${plan}" == "recipe_best" && \
                      "${variant}" == "F16F16" ]]; then
                    rows="${f16_best_rows}"
                    contexts="${f16_best_contexts}"
                elif [[ "${plan}" == "recipe_best" && \
                        "${variant}" == "W4F16" ]]; then
                    rows="${w4_best_rows}"
                    contexts="${w4_best_contexts}"
                fi
                if [[ "${variant}" == "W4U8" ]]; then
                    schedule=control
                else
                    schedule=candidate
                fi
                EXP0084_FP16_NORM_ROWS="${rows}" \
                EXP0084_FP16_NORM_CONTEXTS="${contexts}" \
                    "${project_root}/scripts/run_exp0084.sh" \
                        "${variant}" "${schedule}" "${repeat}" on off \
                        >> "${output_dir}/${plan}/${variant,,}_r${repeat}.jsonl"
            done
        done
    done
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_after.txt"
cmp "${output_dir}/boot_id_before.txt" \
    "${output_dir}/boot_id_after.txt"
printf 'OUTPUT_DIR=%s\n' "${output_dir}"
