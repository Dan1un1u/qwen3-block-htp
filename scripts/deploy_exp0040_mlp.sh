#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0040-mlp}"
package="${QBH_EXP0040_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/mlp_package_layer14_m64}"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_mlp_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}" \
                "${package}/manifest.json"; do
    [[ -f "${artifact}" ]] || {
        printf 'missing EXP-0040 artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}"
"${adb_exe}" push "$(wslpath -w "${host_executable}")" \
    "${remote_root}/qwen3_mlp_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${host_stub}")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${dsp_skel}")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" shell "rm -rf ${remote_root}/mlp_package_layer14_m64 && mkdir -p ${remote_root}/mlp_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${package}")/." \
    "${remote_root}/mlp_package_layer14_m64/" >/dev/null
"${adb_exe}" shell \
    "chmod 755 ${remote_root}/qwen3_mlp_cli ${remote_root}/libqwen3_probe.so ${remote_root}/libqwen3_probe_skel.so && rm -f ${remote_root}/qwen3_mlp_cli.farf"

printf 'REMOTE_ROOT=%s\n' "${remote_root}"
