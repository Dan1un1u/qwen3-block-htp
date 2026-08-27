#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0006}"
storage="${1:-packed_w4u8}"
projection="${2:-gate_up}"
pattern="${3:-identity}"
repeat_count="${4:-1}"
physical_plan="${5:-chunked_hvx1}"
remote_log="${remote_root}/run_${storage}_${projection}_${pattern}_${repeat_count}_${physical_plan}.log"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "ps -A | grep '[q]wen3_probe_cli' || true"
"${adb_exe}" shell "if [ -f ${remote_log} ]; then cat ${remote_log}; fi"
