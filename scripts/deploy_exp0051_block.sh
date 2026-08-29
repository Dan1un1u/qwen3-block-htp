#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package_dir="${QBH_EXP0051_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
remote_root="${EXP0051_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0051}"

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${package_dir}/manifest.json"; do
    [[ -f "${artifact}" ]] || {
        printf 'missing EXP-0051 artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${package_dir}")/." \
    "${remote_root}/block_package_layer14_m64/" >/dev/null
"${adb_exe}" shell \
    "chmod 755 ${remote_root}/qwen3_block_cli ${remote_root}/libqwen3_probe.so ${remote_root}/libqwen3_probe_skel.so"

printf 'REMOTE_ROOT=%s\n' "${remote_root}"
