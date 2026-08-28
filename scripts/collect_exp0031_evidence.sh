#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0031}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0031/artifacts}"
package_dir="${EXP0022_PACKAGE_DIR:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}"
artifact_dir="${artifact_root}/${source_short}"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}/static" "${artifact_dir}"

"${project_root}/scripts/build_exp0031.sh" > "${result_dir}/build.log" 2>&1
QBH_STATIC_OUTPUT_DIR="${result_dir}/static" \
    "${project_root}/scripts/check_exp0031_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0022_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"
"${project_root}/scripts/run_exp0022_vtcm_gate.sh" \
    > "${result_dir}/vtcm_gate.jsonl"

configs=(
    f16_control f16_qk f16_softmax f16_parallel f16_gqa
    w4_control w4_qk w4_softmax w4_parallel w4_gqa
)

set_config() {
    local config="$1"
    case "${config}" in
        f16_control)
            variant=F16F16; workers=2; f16_mode=batch2; w4_mode=control
            attention_mode=control; attention_contexts=1 ;;
        f16_qk)
            variant=F16F16; workers=2; f16_mode=batch2; w4_mode=control
            attention_mode=parallel_qk_norm_rope; attention_contexts=4 ;;
        f16_softmax)
            variant=F16F16; workers=2; f16_mode=batch2; w4_mode=control
            attention_mode=parallel_softmax; attention_contexts=4 ;;
        f16_parallel)
            variant=F16F16; workers=2; f16_mode=batch2; w4_mode=control
            attention_mode=parallel_hvx; attention_contexts=4 ;;
        f16_gqa)
            variant=F16F16; workers=2; f16_mode=batch2; w4_mode=control
            attention_mode=gqa_pipeline; attention_contexts=4 ;;
        w4_control)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_cross
            attention_mode=control; attention_contexts=1 ;;
        w4_qk)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_cross
            attention_mode=parallel_qk_norm_rope; attention_contexts=4 ;;
        w4_softmax)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_cross
            attention_mode=parallel_softmax; attention_contexts=4 ;;
        w4_parallel)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_cross
            attention_mode=parallel_hvx; attention_contexts=4 ;;
        w4_gqa)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_cross
            attention_mode=gqa_pipeline; attention_contexts=4 ;;
        *)
            printf 'unknown config: %s\n' "${config}" >&2
            return 1 ;;
    esac
}

run_config() {
    local config="$1"
    local repeat="$2"
    local audit="$3"
    set_config "${config}"
    "${project_root}/scripts/run_exp0031_block.sh" \
        "${variant}" "${repeat}" on "${audit}" fused \
        "${f16_mode}" hvx "${workers}" 32 "${w4_mode}" hvx \
        streaming 4 64 "${attention_mode}" "${attention_contexts}"
}

for config in "${configs[@]}"; do
    run_config "${config}" 1 on \
        > "${result_dir}/correctness_${config}.jsonl"
    : > "${result_dir}/timing_${config}_repeat1.jsonl"
    : > "${result_dir}/timing_${config}_repeat10.jsonl"
done

for round in 1 2 3 4 5; do
    if (( round % 2 == 1 )); then
        order=("${configs[@]}")
    else
        order=(
            w4_gqa w4_parallel w4_softmax w4_qk w4_control
            f16_gqa f16_parallel f16_softmax f16_qk f16_control
        )
    fi
    for config in "${order[@]}"; do
        run_config "${config}" 1 off \
            >> "${result_dir}/timing_${config}_repeat1.jsonl"
        run_config "${config}" 10 off \
            >> "${result_dir}/timing_${config}_repeat10.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

cp "${package_dir}/manifest.json" "${result_dir}/block_package_manifest.json"
python3 "${project_root}/scripts/validate_exp0031_evidence.py" \
    "${result_dir}" "${package_dir}" > "${result_dir}/matrix_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_vtcm_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"

{
    printf 'experiment=EXP-0031\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'configs=%s\n' "$(IFS=,; printf '%s' "${configs[*]}")"
    printf 'correctness_numerical_audit=on\n'
    printf 'performance_numerical_audit=off\n'
    printf 'attention_modes=control,parallel_qk_norm_rope,parallel_softmax,parallel_hvx,gqa_pipeline\n'
    printf 'attention_hvx_contexts=4_for_candidates\n'
    printf 'hmx_ownership=single_serial_owner\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'physical_contract=fixed_8mib_prepared_session_vtcm\n'
    printf 'intermediate_ddr_allowed=false\n'
    printf 'parent_package_dir=%s\n' "${package_dir}"
    printf 'result_dir=%s\n' "${result_dir}"
    printf 'artifact_dir=%s\n' "${artifact_dir}"
} > "${result_dir}/manifest.txt"

(
    cd "${artifact_dir}"
    sha256sum qwen3_block_cli qwen3_vtcm_cli \
        libqwen3_probe.so libqwen3_probe_skel.so
) > "${result_dir}/artifact_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    "${result_dir}/static"/*.txt > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
