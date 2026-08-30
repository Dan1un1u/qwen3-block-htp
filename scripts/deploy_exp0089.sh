#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0089_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
remote_root="${EXP0089_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0089}"

artifacts=(
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
)
for artifact in "${artifacts[@]}"; do
    if [[ ! -f "${artifact}" ]]; then
        printf 'missing EXP-0089 artifact: %s\n' "${artifact}" >&2
        exit 1
    fi
done
if [[ ! -f "${package}/manifest.json" ]]; then
    printf 'missing package: %s\n' "${package}" >&2
    exit 1
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${artifacts[0]}")" \
    "${remote_root}/qwen3_block_cli"
"${adb_exe}" push "$(wslpath -w "${artifacts[1]}")" \
    "${remote_root}/libqwen3_probe.so"
"${adb_exe}" push "$(wslpath -w "${artifacts[2]}")" \
    "${remote_root}/libqwen3_probe_skel.so"
"${adb_exe}" push "$(wslpath -w "${package}")/." \
    "${remote_root}/block_package_layer14_m64/"
"${adb_exe}" shell "chmod 0755 ${remote_root}/qwen3_block_cli"
printf 'REMOTE_ROOT=%s\n' "${remote_root}"
