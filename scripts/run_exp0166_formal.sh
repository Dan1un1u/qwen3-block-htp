#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${1:?formal result directory required}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
regression_root="${EXP0166_REGRESSION_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0166-regression-w4f16}"
regression_source="/data/local/tmp/qwen3-block-htp/exp0158-w4f16"
package_name="block_package_layer14_m64"
recipe_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
runtime_args="4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"

mkdir -p "${result_dir}/raw"
for round in $(seq 1 10); do
    printf -v index '%02d' "${round}"
    if (( round % 2 == 1 )); then
        order=(control candidate)
    else
        order=(candidate control)
    fi
    for cell in "${order[@]}"; do
        if [[ "${cell}" == control ]]; then
            mode=4
        else
            mode=7
        fi
        QBH_GENERATION_MODE="${mode}" \
            "${project_root}/scripts/run_exp0166.sh" \
            > "${result_dir}/raw/round_${index}_${cell}.jsonl" 2>&1
    done
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "test -d ${regression_source}/${package_name}"
"${adb_exe}" shell "mkdir -p ${regression_root}"
if ! "${adb_exe}" shell "test -d ${regression_root}/${package_name}"; then
    "${adb_exe}" shell \
        "mkdir -p ${regression_root}/${package_name} && cp -as ${regression_source}/${package_name}/. ${regression_root}/${package_name}/"
fi
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
    "${regression_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
    "${regression_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so")" \
    "${regression_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" shell "chmod 755 ${regression_root}/qwen3_block_cli"
"${adb_exe}" shell \
    "cd ${regression_root} && ${recipe_env} QBH_GENERATION_SEQUENCE=0 QBH_REPLAY_DECODE_STEPS=8 QBH_KV_CACHE_LAYOUT=hmx_native_f16 QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=72 LD_LIBRARY_PATH=${regression_root} DSP_LIBRARY_PATH=${regression_root} ADSP_LIBRARY_PATH=${regression_root} ./qwen3_block_cli ${regression_root}/${package_name} W4F16 1 ${runtime_args}" \
    > "${result_dir}/w4f16_exp0158_regression.log" 2>&1

printf 'RESULT_DIR=%s\n' "${result_dir}"
