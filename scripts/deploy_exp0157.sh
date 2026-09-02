#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${1:?package directory required}"
remote_root="${EXP0157_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0157-prefill-carrier-reuse}"
stage_root="${QBH_EXP0157_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0157}"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}" \
                "${package}/manifest.json"; do
    test -f "${artifact}"
done

# Windows adb cannot open WSL UNC paths reliably. Keep the authoritative
# build on WSL ext4 and stage only the three deployable binaries on D:.
mkdir -p "${stage_root}"
cp "${host_executable}" "${stage_root}/qwen3_block_cli"
cp "${host_stub}" "${stage_root}/libqwen3_probe.so"
cp "${dsp_skel}" "${stage_root}/libqwen3_probe_skel.so"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${stage_root}/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
if [[ "${QBH_DEPLOY_BINARIES_ONLY:-0}" != 1 ]]; then
    "${adb_exe}" push "$(wslpath -w "${package}")/." \
        "${remote_root}/block_package_layer14_m64/" >/dev/null
fi
"${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"
