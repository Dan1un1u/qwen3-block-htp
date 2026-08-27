#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0004}"
projection="${1:-gate_up}"
pattern="${2:-identity}"
repeat_count="${3:-1}"

host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_probe_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    if [[ ! -f "${artifact}" ]]; then
        printf 'missing build artifact: %s\n' "${artifact}" >&2
        exit 1
    fi
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}"
"${adb_exe}" push "$(wslpath -w "${host_executable}")" \
    "${remote_root}/qwen3_probe_cli" >/dev/null 2>&1
"${adb_exe}" push "$(wslpath -w "${host_stub}")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null 2>&1
"${adb_exe}" push "$(wslpath -w "${dsp_skel}")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null 2>&1
"${adb_exe}" shell \
    "chmod 755 ${remote_root}/qwen3_probe_cli ${remote_root}/libqwen3_probe.so ${remote_root}/libqwen3_probe_skel.so"

if [[ "${QBH_DIAGNOSTIC_FARF:-0}" == "1" ]]; then
    "${adb_exe}" shell "echo 0x1f > ${remote_root}/qwen3_probe_cli.farf"
else
    "${adb_exe}" shell "rm -f ${remote_root}/qwen3_probe_cli.farf"
fi

if [[ "${QBH_DETACH:-0}" == "1" ]]; then
    remote_log="${remote_root}/run_${projection}_${pattern}_${repeat_count}.log"
    "${adb_exe}" shell "rm -f ${remote_log}"
    "${adb_exe}" shell \
        "cd ${remote_root} && (LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_probe_cli ${projection} ${pattern} ${repeat_count} > ${remote_log} 2>&1 < /dev/null &)"
    printf 'DETACHED_LOG=%s\n' "${remote_log}"
    exit 0
fi

"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_probe_cli ${projection} ${pattern} ${repeat_count}"
