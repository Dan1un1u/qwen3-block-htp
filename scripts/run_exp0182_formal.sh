#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
source_commit="${2:?source commit required}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0173_OVERLAY:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0169/w4u8_greedy193_overlay}"
transformer_package="${QBH_EXP0173_TRANSFORMER_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/candidate_segmented_capacity257}"

if ! git -C "${project_root}" cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
    printf 'source commit does not exist: %s\n' "${source_commit}" >&2
    exit 2
fi
actual_head="$(git -C "${project_root}" rev-parse HEAD)"
if [[ "${source_commit}" != "${actual_head}" ]]; then
    printf 'source commit mismatch: requested=%s HEAD=%s\n' \
        "${source_commit}" "${actual_head}" >&2
    exit 2
fi

mkdir -p "${result_dir}/raw" "${result_dir}/audit"
"${project_root}/scripts/check_exp0182_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0182.sh" \
    > "${result_dir}/deploy.log" 2>&1

for cell in control quartet; do
    QBH_EXP0173_GENERATION_STEPS=34 \
        "${project_root}/scripts/run_exp0182.sh" "${cell}" \
        > "${result_dir}/boundary_${cell}.log" 2>&1
done

QBH_EXP0173_GENERATION_STEPS=193 \
QBH_GENERATION_BOUNDARY_AUDIT=1 \
QBH_GENERATION_AUDIT_DIR="${result_dir}/audit" \
    "${project_root}/scripts/run_exp0182.sh" quartet \
    > "${result_dir}/audit/capture.log" 2>&1

"${python_exe}" "${project_root}/scripts/verify_exp0167_generation.py" \
    --audit-dir "${result_dir}/audit" \
    --package "${package}" \
    --transformer-package "${transformer_package}" \
    --steps 193 --experiment-record 182 --experiment-label EXP-0182 \
    > "${result_dir}/audit/verify.log" 2>&1

QBH_EXP0173_GENERATION_STEPS=193 \
    "${project_root}/scripts/run_exp0182.sh" quartet_poison \
    > "${result_dir}/padding_poison.log" 2>&1

for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    if ((round % 2 == 1)); then
        order=(control quartet)
    else
        order=(quartet control)
    fi
    for cell in "${order[@]}"; do
        QBH_EXP0173_GENERATION_STEPS=193 \
        QBH_GENERATION_BOUNDARY_AUDIT=0 \
            "${project_root}/scripts/run_exp0182.sh" "${cell}" \
            > "${result_dir}/raw/pair_${index}_${cell}.log" 2>&1
    done
done

remote_root=/data/local/tmp/qwen3-block-htp/exp0163-candidate
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
"${project_root}/scripts/run_exp0163.sh" candidate replay \
    > "${result_dir}/w4u8_exp0163_regression.log" 2>&1

"${python_exe}" "${project_root}/scripts/summarize_exp0182.py" \
    --result-dir "${result_dir}" --source-commit "${source_commit}" \
    --rounds 10 --steps 193 --formal
printf 'RESULT_DIR=%s\n' "${result_dir}"
