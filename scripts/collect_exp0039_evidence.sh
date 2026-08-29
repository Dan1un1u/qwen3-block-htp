#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0039}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0039/artifacts}"
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
mkdir -p "${result_dir}" "${artifact_dir}"

"${project_root}/scripts/build_exp0039.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0039_static.sh" \
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

modes=(control overlay deep)

runtime_mode() {
    case "$1" in
        control) printf 'control\n' ;;
        overlay) printf 'phase_overlay\n' ;;
        deep) printf 'gate_up_deep\n' ;;
        *) printf 'unknown EXP-0039 mode: %s\n' "$1" >&2; return 1 ;;
    esac
}

run_config() {
    local mode="$1"
    local repeat="$2"
    local audit="$3"
    "${project_root}/scripts/run_exp0039_block.sh" \
        W4F16 "${repeat}" on "${audit}" "$(runtime_mode "${mode}")"
}

for mode in "${modes[@]}"; do
    printf 'correctness %s\n' "${mode}"
    run_config "${mode}" 1 on > "${result_dir}/correctness_${mode}.jsonl"
    : > "${result_dir}/timing_${mode}_repeat1.jsonl"
    : > "${result_dir}/timing_${mode}_repeat10.jsonl"
done

for round in 1 2 3 4 5 6 7 8 9 10 11; do
    printf 'timing matrix round %d/11\n' "${round}"
    if (( round % 2 == 1 )); then
        order=(control overlay deep)
    else
        order=(deep overlay control)
    fi
    for mode in "${order[@]}"; do
        run_config "${mode}" 1 off \
            >> "${result_dir}/timing_${mode}_repeat1.jsonl"
        run_config "${mode}" 10 off \
            >> "${result_dir}/timing_${mode}_repeat10.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

cp "${package_dir}/manifest.json" "${result_dir}/block_package_manifest.json"
python3 "${project_root}/scripts/validate_exp0039_evidence.py" \
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
    printf 'experiment=EXP-0039\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'variant=W4F16\n'
    printf 'vtcm_plan_modes=control,phase_overlay,gate_up_deep\n'
    printf 'correctness_numerical_audit=on\n'
    printf 'performance_numerical_audit=off\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'timing_rounds=11\n'
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
find "${result_dir}" -maxdepth 1 -type f \
    \( -name '*.jsonl' -o -name '*.json' \) -print0 \
    | sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
