#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0040-mlp}"
storage="${1:-packed_w4}"
repeat_count="${2:-1}"
self_test="${3:-0}"
gate_up_workers="${4:-2}"

if [[ "${DEPLOY_EXP0040:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0040_mlp.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_mlp_cli && test -f ${remote_root}/mlp_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_mlp_cli ${remote_root}/mlp_package_layer14_m64 ${storage} ${repeat_count} ${self_test} ${gate_up_workers}"
