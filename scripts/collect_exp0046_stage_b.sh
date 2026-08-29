#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0046_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0046_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0046}"
artifact_root="${QBH_EXP0046_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0046/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_stage_b"
artifact_dir="${artifact_root}/${source_short}/stage_b"
paired_rounds=7
modes=(w4u8_mlp_input w4u8_mlp_io)

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"

"${project_root}/scripts/build_exp0046.sh" \
    > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0046_stage_b_static.sh" \
    > "${result_dir}/static_gate.json"
QBH_EXP0046_PACKAGE="${package}" \
    "${project_root}/scripts/deploy_exp0046_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"

QBH_EXP0046_PACKAGE="${package}" \
QBH_EXP0046_BOUNDARY_MODE=w4u8_mlp_io \
    "${project_root}/scripts/audit_exp0046_attention.sh" \
    "${result_dir}/attention_implementation_audit" \
    > "${result_dir}/attention_audit.log"

for repeat in 1 10; do
    for mode in "${modes[@]}"; do
        : > "${result_dir}/paired_${mode}_r${repeat}.jsonl"
    done
done

for round in $(seq 1 "${paired_rounds}"); do
    printf 'EXP-0046 Stage B paired round %d/%d\n' \
        "${round}" "${paired_rounds}"
    if (( round % 2 == 1 )); then
        repeat_order=(1 10)
        mode_order=(w4u8_mlp_io w4u8_mlp_input)
    else
        repeat_order=(10 1)
        mode_order=(w4u8_mlp_input w4u8_mlp_io)
    fi
    for repeat in "${repeat_order[@]}"; do
        for mode in "${mode_order[@]}"; do
            "${project_root}/scripts/run_exp0046_block.sh" \
                "${mode}" "${repeat}" on off \
                >> "${result_dir}/paired_${mode}_r${repeat}.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
python3 "${project_root}/scripts/validate_exp0046_stage_b.py" \
    "${result_dir}" "${package}" > "${result_dir}/gate_summary.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"

{
    printf 'experiment=EXP-0046\n'
    printf 'stage=B\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'control=native_mlp_input_plus_row_major_down_unpack\n'
    printf 'candidate=native_mlp_input_plus_native_down_to_final_residual\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'paired_rounds=%d\n' "${paired_rounds}"
    printf 'physical_contract=exact_8mib_vtcm_one_fastrpc_one_hmx_owner\n'
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
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
