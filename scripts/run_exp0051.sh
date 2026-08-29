#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0051_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0051}"
mode="${1:-control}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"

case "${mode}" in
    control) qkvo_mode=qkvo_batch4 ;;
    q_only) qkvo_mode=q_headpair ;;
    full_qkv) qkvo_mode=qkv_headpair ;;
    *)
        printf 'usage: %s [control|q_only|full_qkv] [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} 2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused serial control hvx w4u8_streaming 3 64 u8_log2_gqa_qkv_overlap 4 w4u8_mlp_io_qkv_o ${qkvo_mode}"
