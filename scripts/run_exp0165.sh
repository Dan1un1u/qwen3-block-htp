#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0165_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0165-w4f16}"
generation_mode="${QBH_GENERATION_MODE:-4}"
recipe_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
runtime_args="4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"

if [[ "${generation_mode}" != 1 && "${generation_mode}" != 2 && \
      "${generation_mode}" != 3 && "${generation_mode}" != 4 ]]; then
    printf 'QBH_GENERATION_MODE must be 1 (control), 2 (HVX argmax), 3 (batch4), or 4 (batch8)\n' >&2
    exit 2
fi
if [[ "${QBH_EXP0165_DEPLOY:-0}" == 1 ]]; then
    "${project_root}/scripts/deploy_exp0165.sh"
fi
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root} && ${recipe_env} QBH_GENERATION_SEQUENCE=${generation_mode} QBH_KV_CACHE_LAYOUT=hmx_native_f16 QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=80 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4F16 1 ${runtime_args}"
