#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0040}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/artifacts}"
package="${QBH_EXP0040_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/mlp_package_layer14_m64_output_requant_v5_round}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_stage_a"
artifact_dir="${artifact_root}/${source_short}"

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
QBH_EXP0040_PACKAGE="${package}" \
    "${project_root}/scripts/deploy_exp0040_mlp.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

run_case() {
    local storage="$1"
    local repeat="$2"
    local self_test="$3"
    local workers="$4"
    "${project_root}/scripts/run_exp0040_mlp.sh" \
        "${storage}" "${repeat}" "${self_test}" "${workers}"
}

for workers in 2 3; do
    for storage in packed_w4 expanded_s8; do
        key=packed
        [[ "${storage}" == expanded_s8 ]] && key=expanded
        run_case "${storage}" 1 1 "${workers}" \
            > "${result_dir}/correctness_w${workers}_${key}.jsonl"
        : > "${result_dir}/timing_w${workers}_${key}_r1.jsonl"
        : > "${result_dir}/timing_w${workers}_${key}_r10.jsonl"
    done
done

for round in 1 2 3 4 5 6 7 8 9 10 11; do
    printf 'EXP-0040 Stage-A timing round %d/11\n' "${round}"
    if (( round % 2 == 1 )); then
        worker_order=(2 3)
        repeat_order=(1 10)
        storage_order=(packed_w4 expanded_s8)
    else
        worker_order=(3 2)
        repeat_order=(10 1)
        storage_order=(expanded_s8 packed_w4)
    fi
    for workers in "${worker_order[@]}"; do
        for repeat in "${repeat_order[@]}"; do
            for storage in "${storage_order[@]}"; do
                key=packed
                [[ "${storage}" == expanded_s8 ]] && key=expanded
                run_case "${storage}" "${repeat}" 0 "${workers}" \
                    >> "${result_dir}/timing_w${workers}_${key}_r${repeat}.jsonl"
            done
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
python3 "${project_root}/scripts/audit_exp0040_package.py" "${package}" \
    > "${result_dir}/package_numerical_audit.json"
python3 "${project_root}/scripts/validate_exp0040_stage_a.py" \
    "${result_dir}" "${package}" > "${result_dir}/stage_a_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_mlp_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"

{
    printf 'experiment=EXP-0040\n'
    printf 'stage=A\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_m64_mlp\n'
    printf 'activation=asymmetric_U8\n'
    printf 'variants=packed_W4,Expanded_S8\n'
    printf 'contract_gate_up_workers=2\n'
    printf 'diagnostic_gate_up_workers=3\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'paired_rounds=11\n'
    printf 'physical_contract=fixed_8mib_prepared_session_vtcm\n'
    printf 'intermediate_ddr_allowed=false\n'
    printf 'package=%s\n' "${package}"
    printf 'result_dir=%s\n' "${result_dir}"
    printf 'artifact_dir=%s\n' "${artifact_dir}"
} > "${result_dir}/manifest.txt"

(
    cd "${artifact_dir}"
    sha256sum qwen3_mlp_cli libqwen3_probe.so libqwen3_probe_skel.so
) > "${result_dir}/artifact_sha256.txt"
sha256sum "${package}"/* > "${result_dir}/package_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
