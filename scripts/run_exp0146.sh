#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0146_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0146-w4f16}"
cell="${1:-control}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
qkv_schedule=control

case "${cell}" in
    control)
        ;;
    candidate)
        qkv_schedule=cross_projection_ring
        ;;
    *)
        printf 'usage: %s control|candidate [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

runtime_args="4 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && QBH_QKV_SCHEDULE=${qkv_schedule} QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4F16 ${repeat_count} ${runtime_args}"
