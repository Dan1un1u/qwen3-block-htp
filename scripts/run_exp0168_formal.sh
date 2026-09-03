#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"

mkdir -p "${result_dir}/raw" "${result_dir}/audit"
"${project_root}/scripts/check_exp0168_static.sh" \
    > "${result_dir}/static_gate.json"

QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/audit" \
QBH_EXP0168_GENERATION_MODE=9 \
    "${project_root}/scripts/run_exp0168.sh" \
    > "${result_dir}/audit/capture.log" 2>&1

"${python_exe}" "${project_root}/scripts/verify_exp0167_generation.py" \
    --audit-dir "${result_dir}/audit" \
    --experiment-record 168 --experiment-label EXP-0168 \
    > "${result_dir}/audit/verify.log" 2>&1

for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    if (( round % 2 == 1 )); then
        order=(8 9)
    else
        order=(9 8)
    fi
    for mode in "${order[@]}"; do
        label=control
        if [[ "${mode}" == 9 ]]; then
            label=candidate
        fi
        QBH_GENERATION_BOUNDARY_AUDIT=0 \
        QBH_EXP0168_GENERATION_MODE="${mode}" \
            "${project_root}/scripts/run_exp0168.sh" \
            > "${result_dir}/raw/${label}_${index}.log" 2>&1
    done
done

for remote_root in /data/local/tmp/qwen3-block-htp/exp0163-candidate; do
    "${adb_exe}" push \
        "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
        "${remote_root}/qwen3_block_cli" >/dev/null
    "${adb_exe}" push \
        "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
        "${remote_root}/libqwen3_probe.so" >/dev/null
    "${adb_exe}" push \
        "$(wslpath -w "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so")" \
        "${remote_root}/libqwen3_probe_skel.so" >/dev/null
    "${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"
done
"${project_root}/scripts/run_exp0163.sh" candidate replay \
    > "${result_dir}/w4u8_exp0163_regression.log" 2>&1

printf 'RESULT_DIR=%s\n' "${result_dir}"
