#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0040}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/stage_b_artifacts}"
package="${QBH_EXP0040_STAGE_B_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/block_package_layer14_m64_stage_b}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_stage_b"
artifact_dir="${artifact_root}/${source_short}"
paired_rounds=7

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"

"${project_root}/scripts/build_exp0040.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0040_static.sh" \
    > "${result_dir}/static_gate.json"
QBH_EXP0040_STAGE_B_PACKAGE="${package}" \
    "${project_root}/scripts/deploy_exp0040_stage_b_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

run_w4u8() {
    local repeat="$1"
    local mode="$2"
    local workers=1
    [[ "${mode}" == w4u8_streaming ]] && workers=3
    "${project_root}/scripts/run_exp0040_stage_b_block.sh" \
        W4U8 "${repeat}" "${mode}" "${workers}" on off
}

run_w4f16() {
    local repeat="$1"
    "${project_root}/scripts/run_exp0040_stage_b_block.sh" \
        W4F16 "${repeat}" control 1 on off
}

for repeat in 1 10; do
    : > "${result_dir}/paired_w4u8_control_r${repeat}.jsonl"
    : > "${result_dir}/paired_w4u8_candidate_r${repeat}.jsonl"
    : > "${result_dir}/w4f16_reference_r${repeat}.jsonl"
done

for round in $(seq 1 "${paired_rounds}"); do
    printf 'EXP-0040 Stage-B paired round %d/%d\n' \
        "${round}" "${paired_rounds}"
    if (( round % 2 == 1 )); then
        repeat_order=(1 10)
        mode_order=(control w4u8_streaming)
    else
        repeat_order=(10 1)
        mode_order=(w4u8_streaming control)
    fi
    for repeat in "${repeat_order[@]}"; do
        for mode in "${mode_order[@]}"; do
            key=control
            [[ "${mode}" == w4u8_streaming ]] && key=candidate
            run_w4u8 "${repeat}" "${mode}" \
                >> "${result_dir}/paired_w4u8_${key}_r${repeat}.jsonl"
        done
        run_w4f16 "${repeat}" \
            >> "${result_dir}/w4f16_reference_r${repeat}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
python3 "${project_root}/scripts/validate_exp0040_stage_b.py" \
    "${result_dir}" "${package}" > "${result_dir}/stage_b_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"

{
    printf 'experiment=EXP-0040\n'
    printf 'stage=B\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'control=historical_complete_block_W4U8\n'
    printf 'candidate=streaming_packed_W4U8_MLP\n'
    printf 'characterization=W4F16_EXP0038_NORMS\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'paired_rounds=%d\n' "${paired_rounds}"
    printf 'physical_contract=exact_8mib_prepared_session_vtcm\n'
    printf 'intermediate_ddr_allowed=false\n'
    printf 'package=%s\n' "${package}"
    printf 'result_dir=%s\n' "${result_dir}"
    printf 'artifact_dir=%s\n' "${artifact_dir}"
} > "${result_dir}/manifest.txt"

(
    cd "${artifact_dir}"
    sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so
) > "${result_dir}/artifact_sha256.txt"
sha256sum "${package}"/* > "${result_dir}/package_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
