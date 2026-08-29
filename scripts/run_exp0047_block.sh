#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0047-block}"
variant_key="${1:-w4f16}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"

if [[ "${DEPLOY_EXP0047:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0047_block.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"

case "${variant_key}" in
    w4f16)
        args="W4F16 ${repeat_count} 3 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross_prefetch combined_hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms"
        ;;
    w4u8)
        args="W4U8 ${repeat_count} 2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused serial control hvx w4u8_streaming 3 64 u8_log2_gqa_qkv_overlap 4 w4u8_mlp_io qkvo_batch4"
        ;;
    *)
        printf 'usage: %s [w4f16|w4u8] [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${args}"
