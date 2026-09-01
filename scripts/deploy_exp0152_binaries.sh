#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
artifacts=(
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
)

for artifact in "${artifacts[@]}"; do
    test -f "${artifact}"
done
"${adb_exe}" get-state >/dev/null
for recipe in f16f16 w4f16 w4u8; do
    remote_root="/data/local/tmp/qwen3-block-htp/exp0152-${recipe}"
    "${adb_exe}" push "$(wslpath -w "${artifacts[0]}")" \
        "${remote_root}/qwen3_block_cli" >/dev/null
    "${adb_exe}" push "$(wslpath -w "${artifacts[1]}")" \
        "${remote_root}/libqwen3_probe.so" >/dev/null
    "${adb_exe}" push "$(wslpath -w "${artifacts[2]}")" \
        "${remote_root}/libqwen3_probe_skel.so" >/dev/null
    "${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"
done
