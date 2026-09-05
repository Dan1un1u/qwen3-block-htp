#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${QBH_EXP0214_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0214-m64-direct-w4}"
package="${QBH_EXP0214_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0187/real_layer14_m64}"

host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_probe_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}" "${package}/manifest.json"; do
    test -f "${artifact}" || { printf 'missing artifact: %s\n' "${artifact}" >&2; exit 1; }
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/package_hmxref"
for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    "${adb_exe}" push "$(wslpath -w "${artifact}")" "${remote_root}/" >/dev/null
done
"${adb_exe}" push "$(wslpath -w "${package}")/." "${remote_root}/package_hmxref/" >/dev/null
"${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_probe_cli"
