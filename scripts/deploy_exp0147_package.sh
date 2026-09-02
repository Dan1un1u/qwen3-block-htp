#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${1:?package directory required}"
remote_root="${EXP0147_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0147}"
artifacts=(
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
)

for artifact in "${artifacts[@]}" "${package}/manifest.json"; do
    test -f "${artifact}"
done
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${artifacts[0]}")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${artifacts[1]}")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${artifacts[2]}")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
if [[ "${QBH_DEPLOY_BINARIES_ONLY:-0}" != 1 ]]; then
    "${adb_exe}" push "$(wslpath -w "${package}")/." \
        "${remote_root}/block_package_layer14_m64/" >/dev/null
fi
"${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"
