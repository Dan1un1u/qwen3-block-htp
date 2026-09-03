#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
source_commit="${2:?source commit required}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0173_OVERLAY:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0169/w4u8_greedy193_overlay}"
transformer_package="${QBH_EXP0173_TRANSFORMER_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/candidate_segmented_capacity257}"

mkdir -p "${result_dir}/raw" "${result_dir}/audit"
"${project_root}/scripts/check_exp0173_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0173.sh" \
    > "${result_dir}/deploy.log" 2>&1

QBH_EXP0173_LM_HEAD_GROUP_TILES=16 \
QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/audit" \
    "${project_root}/scripts/run_exp0173.sh" \
    > "${result_dir}/audit/capture.log" 2>&1

"${python_exe}" "${project_root}/scripts/verify_exp0167_generation.py" \
    --audit-dir "${result_dir}/audit" \
    --package "${package}" \
    --transformer-package "${transformer_package}" \
    --steps 193 --experiment-record 173 --experiment-label EXP-0173 \
    > "${result_dir}/audit/verify.log" 2>&1

for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    if (( round % 2 == 1 )); then
        order=(8 16)
    else
        order=(16 8)
    fi
    for group_tiles in "${order[@]}"; do
        QBH_EXP0173_LM_HEAD_GROUP_TILES="${group_tiles}" \
        QBH_GENERATION_BOUNDARY_AUDIT=0 \
            "${project_root}/scripts/run_exp0173.sh" \
            > "${result_dir}/raw/pair_${index}_batch${group_tiles}.log" 2>&1
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

"${python_exe}" "${project_root}/scripts/summarize_exp0173.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}"
printf 'RESULT_DIR=%s\n' "${result_dir}"
