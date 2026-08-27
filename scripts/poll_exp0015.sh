#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0015}"
storage="${1:-packed_w4_hmx_postscale}"
projection="${2:-gate_up}"
pattern="${3:-identity}"
repeat_count="${4:-1}"
physical_plan="${5:-slots8e7_chunk64_dma_chain4}"
output_assembly="${6:-linked_2d_dma}"
resource_lifetime="${7:-prepared_session}"
remote_log="${remote_root}/run_${storage}_${projection}_${pattern}_${repeat_count}_${physical_plan}_${output_assembly}_${resource_lifetime}.log"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "ps -A | grep '[q]wen3_probe_cli' || true"
"${adb_exe}" shell "if [ -f ${remote_log} ]; then cat ${remote_log}; fi"
