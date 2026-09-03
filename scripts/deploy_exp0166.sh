#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0166_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0166-w4f16}"
source_root="${EXP0166_SOURCE_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0165-w4f16}"
package_name="block_package_layer14_m64"

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"; do
    test -f "${artifact}"
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "test -d ${source_root}/${package_name}"
"${adb_exe}" shell "mkdir -p ${remote_root}"
if ! "${adb_exe}" shell "test -d ${remote_root}/${package_name}"; then
    "${adb_exe}" shell \
        "mkdir -p ${remote_root}/${package_name} && cp -as ${source_root}/${package_name}/. ${remote_root}/${package_name}/"
fi
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"

printf 'REMOTE_ROOT=%s\n' "${remote_root}"
