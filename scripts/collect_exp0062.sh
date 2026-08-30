#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0062_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0062_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0062}"
artifact_root="${QBH_EXP0062_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0062/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
source_branch="$(git -C "${project_root}" branch --show-current)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_formal"
artifact_dir="${artifact_root}/${source_short}/formal"
paired_rounds=7

if ! git -C "${project_root}" diff --quiet || ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"
"${project_root}/scripts/build_exp0062.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0062_static.sh" > "${result_dir}/static_gate.json"
QBH_EXP0062_PACKAGE="${package}" "${project_root}/scripts/deploy_exp0062_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
for mode in control candidate; do
    "${project_root}/scripts/run_exp0062.sh" "${mode}" 1 on on \
        > "${result_dir}/correctness_${mode}.jsonl"
done
for repeat in 1 10; do for mode in control candidate; do
    : > "${result_dir}/paired_${mode}_r${repeat}.jsonl"
done; done
for round in 1 2 3 4 5 6 7; do
    printf 'EXP-0062 paired round %d/%d\n' "${round}" "${paired_rounds}"
    if (( round % 2 == 1 )); then repeats=(1 10); modes=(control candidate)
    else repeats=(10 1); modes=(candidate control); fi
    for repeat in "${repeats[@]}"; do for mode in "${modes[@]}"; do
        "${project_root}/scripts/run_exp0062.sh" "${mode}" "${repeat}" on off \
            >> "${result_dir}/paired_${mode}_r${repeat}.jsonl"
    done; done
done
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
python3 "${project_root}/scripts/validate_exp0062.py" "${result_dir}" "${package}" \
    > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/validate_exp0062.py" "${result_dir}" "${package}" --report \
    > "${result_dir}/full_profiling_report.md"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"
{
    printf 'experiment=EXP-0062\nsource_branch=%s\nsource_head=%s\n' "${source_branch}" "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'control=EXP0061-per-task-rope-fp16-to-sf32\n'
    printf 'candidate=EXP0061-shared-64-row-vtcm-rope-sf32-cache\n'
    printf 'repeat_contract=repeat1,repeat10\npaired_rounds=%d\n' "${paired_rounds}"
    printf 'physical_contract=exact_8mib_vtcm_one_fastrpc_one_hmx_owner\nintermediate_ddr_allowed=false\n'
    printf 'package=%s\nresult_dir=%s\nartifact_dir=%s\n' "${package}" "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
(cd "${artifact_dir}" && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    > "${result_dir}/artifact_sha256.txt"
sha256sum "${package}"/* > "${result_dir}/package_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    "${result_dir}/full_profiling_report.md" > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
