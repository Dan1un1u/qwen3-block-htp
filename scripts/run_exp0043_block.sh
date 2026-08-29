#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0043-block}"
mode="${1:-instrumented}"
repeat_count="${2:-1}"

case "${mode}" in
    instrumented)
        attribution_mode=on
        ;;
    control)
        attribution_mode=off
        ;;
    *)
        printf 'usage: %s [instrumented|control] [repeat]\n' "$0" >&2
        exit 2
        ;;
esac

if [[ "${DEPLOY_EXP0043:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0043_block.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} 2 32 rms_rope_softmax ${attribution_mode} off fused serial control hvx w4u8_streaming 3 64 u8_log2_gqa 4 control"
