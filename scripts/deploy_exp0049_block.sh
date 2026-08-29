#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package_dir="${QBH_EXP0049_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
candidate_root="${EXP0049_CANDIDATE_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0049-candidate}"
control_root="${EXP0049_CONTROL_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0049-control}"
control_artifacts="${QBH_EXP0049_CONTROL_ARTIFACTS:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0048/artifacts/61cd86fdabc3/stage_b}"

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${control_artifacts}/qwen3_block_cli" \
    "${control_artifacts}/libqwen3_probe.so" \
    "${control_artifacts}/libqwen3_probe_skel.so" \
    "${package_dir}/manifest.json"; do
    [[ -f "${artifact}" ]] || {
        printf 'missing EXP-0049 artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

deploy_one() {
    local root="$1"
    local host_cli="$2"
    local host_stub="$3"
    local dsp_skel="$4"
    "${adb_exe}" shell "mkdir -p ${root}/block_package_layer14_m64"
    "${adb_exe}" push "$(wslpath -w "${host_cli}")" \
        "${root}/qwen3_block_cli" >/dev/null
    "${adb_exe}" push "$(wslpath -w "${host_stub}")" \
        "${root}/libqwen3_probe.so" >/dev/null
    "${adb_exe}" push "$(wslpath -w "${dsp_skel}")" \
        "${root}/libqwen3_probe_skel.so" >/dev/null
    "${adb_exe}" push "$(wslpath -w "${package_dir}")/." \
        "${root}/block_package_layer14_m64/" >/dev/null
    "${adb_exe}" shell \
        "chmod 755 ${root}/qwen3_block_cli ${root}/libqwen3_probe.so ${root}/libqwen3_probe_skel.so"
}

"${adb_exe}" get-state >/dev/null
deploy_one "${candidate_root}" \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
deploy_one "${control_root}" \
    "${control_artifacts}/qwen3_block_cli" \
    "${control_artifacts}/libqwen3_probe.so" \
    "${control_artifacts}/libqwen3_probe_skel.so"

printf 'CANDIDATE_REMOTE_ROOT=%s\n' "${candidate_root}"
printf 'CONTROL_REMOTE_ROOT=%s\n' "${control_root}"
