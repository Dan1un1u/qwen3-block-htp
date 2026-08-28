#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0022-block}"
variant="${1:-F16F16}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
residual_mode="${5:-scalar}"
common_ops_mode="${6:-hvx}"
w4f16_hvx_workers="${7:-2}"
w4f16_region_tiles="${8:-16}"

if [[ "${DEPLOY_EXP0026:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0022_block.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${w4f16_hvx_workers} ${w4f16_region_tiles} ${common_ops_mode} ${attribution_mode} ${audit_mode} ${residual_mode}"
