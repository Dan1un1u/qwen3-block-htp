#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0003}"
pattern="${1:-identity}"
repeat_count="${2:-1}"
remote_log="${remote_root}/run_${pattern}_${repeat_count}.log"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "ps -A | grep '[q]wen3_probe_cli' || true"
"${adb_exe}" shell "if [ -f ${remote_log} ]; then cat ${remote_log}; fi"
