#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh"
set -u
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0005}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0005}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}"
artifact_dir="${artifact_root}/${source_short}"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

mkdir -p "${result_dir}/static" "${artifact_dir}"

"${project_root}/scripts/build_exp0005.sh" \
    > "${result_dir}/build.log" 2>&1
QBH_STATIC_OUTPUT_DIR="${result_dir}/static" \
    "${project_root}/scripts/check_exp0005_static.sh" \
    > "${result_dir}/static_gate.json"

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

: > "${result_dir}/correctness_repeat1.jsonl"
: > "${result_dir}/timing_repeat10.jsonl"
for storage in expanded_s8_control packed_w4u8; do
    for projection in gate_up down; do
        for pattern in identity signed structured boundary; do
            "${project_root}/scripts/run_exp0005.sh" \
                "${storage}" "${projection}" "${pattern}" 1 \
                | tee -a "${result_dir}/correctness_repeat1.jsonl"
        done
    done
done
for storage in expanded_s8_control packed_w4u8; do
    for projection in gate_up down; do
        for pattern in identity signed structured boundary; do
            "${project_root}/scripts/run_exp0005.sh" \
                "${storage}" "${projection}" "${pattern}" 10 \
                | tee -a "${result_dir}/timing_repeat10.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/validate_exp0005_evidence.py" \
    "${result_dir}" > "${result_dir}/matrix_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_probe_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"

{
    printf 'experiment=EXP-0005\n'
    printf 'source_branch=%s\n' \
        "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'hexagon_sdk=%s\n' "${HEXAGON_SDK_ROOT:-unset}"
    printf 'hexagon_tools=%s\n' "${DEFAULT_HEXAGON_TOOLS_ROOT:-unset}"
    printf 'android_ndk=%s\n' "${ANDROID_ROOT_DIR:-unset}"
    printf 'artifact_dir=%s\n' "${artifact_dir}"
} > "${result_dir}/manifest.txt"

(cd "${artifact_dir}" && sha256sum qwen3_probe_cli libqwen3_probe.so \
    libqwen3_probe_skel.so) > "${result_dir}/artifact_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    "${result_dir}/static"/*.txt > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
