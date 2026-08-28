#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0036/stage_b}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0036/stage_b_artifacts}"
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

"${project_root}/scripts/build_exp0036.sh" > "${result_dir}/build.log" 2>&1
QBH_STATIC_OUTPUT_DIR="${result_dir}/static" \
    "${project_root}/scripts/check_exp0036_static.sh" \
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

configs=(w4_b4 w4_b8 f16_b4 f16_b8)

run_config() {
    local config="$1"
    local repeat="$2"
    local audit="$3"
    local variant workers f16_mode w4_mode mlp_mode
    case "${config}" in
        w4_b4)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_gate4_dma8_cross
            mlp_mode=crouton_native ;;
        w4_b8)
            variant=W4F16; workers=3; f16_mode=serial
            w4_mode=adaptive_down96_gate4_dma8_cross
            mlp_mode=crouton_native_batch8 ;;
        f16_b4)
            variant=F16F16; workers=2; f16_mode=gate4
            w4_mode=control; mlp_mode=streaming ;;
        f16_b8)
            variant=F16F16; workers=2; f16_mode=gate8
            w4_mode=control; mlp_mode=crouton_native_batch8 ;;
        *) printf 'unknown config: %s\n' "${config}" >&2; return 1 ;;
    esac
    "${project_root}/scripts/run_exp0036_block.sh" \
        "${variant}" "${repeat}" on "${audit}" fused \
        "${f16_mode}" hvx "${workers}" 32 "${w4_mode}" hvx \
        "${mlp_mode}" 4 64 gqa_qkv_overlap 4
}

for config in "${configs[@]}"; do
    run_config "${config}" 1 on \
        > "${result_dir}/correctness_${config}.jsonl"
    : > "${result_dir}/timing_${config}_repeat1.jsonl"
    : > "${result_dir}/timing_${config}_repeat10.jsonl"
done

for round in 1 2 3 4 5; do
    if (( round % 2 == 1 )); then
        order=(w4_b4 w4_b8 f16_b4 f16_b8)
    else
        order=(f16_b8 f16_b4 w4_b8 w4_b4)
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
python3 "${project_root}/scripts/validate_exp0036_stage_b_evidence.py" \
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
    printf 'experiment=EXP-0036\n'
    printf 'stage=B\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'configs=w4_b4,w4_b8,f16_b4,f16_b8\n'
    printf 'correctness_numerical_audit=on\n'
    printf 'performance_numerical_audit=off\n'
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
