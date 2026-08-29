#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package_dir="${QBH_EXP0040_STAGE_B_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/block_package_layer14_m64_stage_b}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0040-stage-b}"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}" \
                "${package_dir}/manifest.json"; do
    [[ -e "${artifact}" ]] || {
        printf 'missing EXP-0040 Stage-B artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${host_executable}")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${host_stub}")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${dsp_skel}")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${package_dir}")/." \
    "${remote_root}/block_package_layer14_m64/" >/dev/null
"${adb_exe}" shell \
    "chmod 755 ${remote_root}/qwen3_block_cli ${remote_root}/libqwen3_probe.so ${remote_root}/libqwen3_probe_skel.so"

printf 'REMOTE_ROOT=%s\n' "${remote_root}"
printf 'REMOTE_PACKAGE=%s\n' "${remote_root}/block_package_layer14_m64"
