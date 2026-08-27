#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh"
set -u
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0010}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0010}"
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

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}/static" "${artifact_dir}"

"${project_root}/scripts/build_exp0010.sh" \
    > "${result_dir}/build.log" 2>&1
QBH_STATIC_OUTPUT_DIR="${result_dir}/static" \
    "${project_root}/scripts/check_exp0010_static.sh" \
    > "${result_dir}/static_gate.json"

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

storages=(
    expanded_s8_control
    packed_w4_hvx_prescale
    packed_w4_hmx_postscale
    packed_w4_hmx_postscale
)
plans=(
    exp0005_full_bundle_control
    slots4_chunk32
    slots4_chunk32
    slots4_chunk64
)
patterns=(identity signed structured boundary)
deployed=0

run_one() {
    local storage="$1"
    local pattern="$2"
    local repeat_count="$3"
    local plan="$4"
    local output="$5"
    if [[ "${deployed}" == "0" ]]; then
        "${project_root}/scripts/run_exp0010.sh" \
            "${storage}" gate_up "${pattern}" "${repeat_count}" "${plan}" \
            | tee -a "${output}"
        deployed=1
    else
        QBH_SKIP_DEPLOY=1 "${project_root}/scripts/run_exp0010.sh" \
            "${storage}" gate_up "${pattern}" "${repeat_count}" "${plan}" \
            | tee -a "${output}"
    fi
}

: > "${result_dir}/correctness_repeat1.jsonl"
for pattern in "${patterns[@]}"; do
    for index in "${!plans[@]}"; do
        run_one "${storages[index]}" "${pattern}" 1 "${plans[index]}" \
            "${result_dir}/correctness_repeat1.jsonl"
    done
done

: > "${result_dir}/timing_repeat10.jsonl"
for round in 0 1 2; do
    for pattern_index in "${!patterns[@]}"; do
        start=$(((round + pattern_index) % ${#plans[@]}))
        for step in "${!plans[@]}"; do
            index=$(((start + step) % ${#plans[@]}))
            run_one "${storages[index]}" "${patterns[pattern_index]}" 10 \
                "${plans[index]}" "${result_dir}/timing_repeat10.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/validate_exp0010_evidence.py" \
    "${result_dir}" > "${result_dir}/matrix_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_probe_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"

{
    printf 'experiment=EXP-0010\n'
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
