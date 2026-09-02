#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
recipe="${1:?f16f16 or w4f16 required}"
layout="${2:-hmx_native_f16}"
stage="${3:-replay}"
remote_root="${EXP0158_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0158-${recipe}}"

case "${recipe}" in
    f16f16)
        variant=F16F16
        recipe_env=""
        runtime_args="2 32 hvx on off fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    w4f16)
        variant=W4F16
        recipe_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
        runtime_args="4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    *) exit 2 ;;
esac
case "${layout}" in row_major|hmx_native_f16) ;; *) exit 2 ;; esac
case "${stage}" in
    layout) stage_env="QBH_LAYOUT_ONLY=1" ;;
    map) stage_env="QBH_MAP_ONLY=1" ;;
    replay) stage_env="" ;;
    *) exit 2 ;;
esac

if [[ "${QBH_EXP0158_DEPLOY:-0}" == 1 ]]; then
    EXP0158_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0158.sh" "${recipe}"
fi
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root} && ${recipe_env} QBH_KV_CACHE_LAYOUT=${layout} ${stage_env} QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=72 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} 1 ${runtime_args}"
