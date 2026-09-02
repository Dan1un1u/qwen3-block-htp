#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
recipe="${1:?f16f16 or w4f16 required}"
package="${2:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0158/${recipe}}"
remote_root="${EXP0158_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0158-${recipe}}"
source_remote="/data/local/tmp/qwen3-block-htp/exp0152-${recipe}/block_package_layer14_m64"
stage_root="${QBH_EXP0158_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0158-${recipe}}"
overlay="${stage_root}/cache_native_f16_overlay.tar"

case "${recipe}" in f16f16|w4f16) ;; *) exit 2 ;; esac
for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${package}/manifest.json"; do
    test -f "${artifact}"
done

mkdir -p "${stage_root}"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${stage_root}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${stage_root}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${stage_root}/"
tar -C "${package}" -cf "${overlay}" manifest.json \
    $(cd "${package}" && find layer* -maxdepth 1 -type f -name '*hmx_f16*' -print | sort)

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "test -d ${source_remote}"
"${adb_exe}" shell "mkdir -p ${remote_root}"
if ! "${adb_exe}" shell "test -d ${remote_root}/block_package_layer14_m64"; then
    "${adb_exe}" shell \
        "mkdir -p ${remote_root}/block_package_layer14_m64 && cp -as ${source_remote}/. ${remote_root}/block_package_layer14_m64/"
fi
"${adb_exe}" push "$(wslpath -w "${stage_root}/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${overlay}")" \
    "${remote_root}/cache_native_f16_overlay.tar" >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root}/block_package_layer14_m64 && rm -f manifest.json && tar -xf ../cache_native_f16_overlay.tar && cd .. && rm -f cache_native_f16_overlay.tar && chmod 755 qwen3_block_cli"
