#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0047_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0047_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0047}"
artifact_root="${QBH_EXP0047_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0047/artifacts}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0047-block}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
source_branch="$(git -C "${project_root}" branch --show-current)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_formal"
artifact_dir="${artifact_root}/${source_short}"
result_dir_windows="$(wslpath -w "${result_dir}")"
artifact_dir_windows="$(wslpath -w "${artifact_dir}")"
paired_rounds=7
variants=(w4f16 w4u8)

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"

"${project_root}/scripts/build_exp0047.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0047_static.sh" > "${result_dir}/static_gate.json"
QBH_EXP0047_PACKAGE="${package}" REMOTE_ROOT="${remote_root}" \
    "${project_root}/scripts/deploy_exp0047_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"

for variant in "${variants[@]}"; do
    REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/run_exp0047_block.sh" \
        "${variant}" 1 on on > "${result_dir}/correctness_${variant}.jsonl"
done

REMOTE_ROOT="${remote_root}" QBH_EXP0047_PACKAGE="${package}" \
    "${project_root}/scripts/audit_exp0047_attention.sh" \
    "${result_dir}/attention_implementation_audit" \
    > "${result_dir}/attention_audit.log"

for repeat in 1 10; do
    for variant in "${variants[@]}"; do
        : > "${result_dir}/paired_${variant}_r${repeat}.jsonl"
    done
done

for round in 1 2 3 4 5 6 7; do
    printf 'EXP-0047 paired round %d/%d\n' "${round}" "${paired_rounds}"
    if (( round % 2 == 1 )); then
        repeat_order=(1 10)
        variant_order=(w4u8 w4f16)
    else
        repeat_order=(10 1)
        variant_order=(w4f16 w4u8)
    fi
    for repeat in "${repeat_order[@]}"; do
        for variant in "${variant_order[@]}"; do
            REMOTE_ROOT="${remote_root}" \
                "${project_root}/scripts/run_exp0047_block.sh" \
                "${variant}" "${repeat}" on off \
                >> "${result_dir}/paired_${variant}_r${repeat}.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"

{
    printf 'experiment=EXP-0047\n'
    printf 'source_branch=%s\n' "${source_branch}"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'control=W4F16-EXP0038-NORMS-current-source\n'
    printf 'candidate=W4U8-EXP0046-NATIVE-MLP-IO\n'
    printf 'comparison_semantics=paired-current-best-not-identical-schedule-single-variable\n'
    printf 'repeat_contract=repeat1,repeat10\n'
    printf 'paired_rounds=%d\n' "${paired_rounds}"
    printf 'physical_contract=exact_8mib_vtcm_one_fastrpc_one_hmx_owner\n'
    printf 'intermediate_ddr_allowed=false\n'
    printf 'package=%s\n' "${package}"
    printf 'result_dir=%s\n' "${result_dir}"
    printf 'result_dir_windows=%s\n' "${result_dir_windows}"
    printf 'artifact_dir=%s\n' "${artifact_dir}"
    printf 'artifact_dir_windows=%s\n' "${artifact_dir_windows}"
} > "${result_dir}/manifest.txt"

python3 "${project_root}/scripts/validate_exp0047_evidence.py" \
    "${result_dir}" "${package}" > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/validate_exp0047_evidence.py" \
    "${result_dir}" "${package}" --report \
    > "${result_dir}/full_profiling_report.md"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"

(
    cd "${artifact_dir}"
    sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so
) > "${result_dir}/artifact_sha256.txt"
sha256sum "${package}"/* > "${result_dir}/package_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    "${result_dir}/full_profiling_report.md" \
    > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
