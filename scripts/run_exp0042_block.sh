#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0042-block}"
mode="${1:-candidate}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"

case "${mode}" in
    candidate)
        attention_pack_mode=hvx
        attention_pipeline_mode=u8_log2_gqa
        attention_hvx_contexts=4
        ;;
    control)
        attention_pack_mode=control
        attention_pipeline_mode=control
        attention_hvx_contexts=1
        ;;
    *)
        printf 'usage: %s [candidate|control] [repeat] [attribution] [audit]\n' \
            "$0" >&2
        exit 2
        ;;
esac

if [[ "${DEPLOY_EXP0042:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0042_block.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} 2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused serial control ${attention_pack_mode} w4u8_streaming 3 64 ${attention_pipeline_mode} ${attention_hvx_contexts} control"
