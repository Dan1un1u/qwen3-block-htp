#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0096_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0096_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0096}"
artifact_root="${QBH_EXP0096_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0096/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_a_${source_short}"
artifact_dir="${artifact_root}/${source_short}/stage_a"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0096 stage-A collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
if [[ -e "${result_dir}" || -e "${artifact_dir}" ]]; then
    printf 'EXP-0096 evidence destination already exists\n' >&2
    exit 1
fi
mkdir -p "${result_dir}" "${artifact_dir}"
"${project_root}/scripts/build_exp0096.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0096_static.sh" \
    > "${result_dir}/static_gate.json"
QBH_EXP0096_PACKAGE="${package}" \
    "${project_root}/scripts/deploy_exp0096_block.sh" \
    > "${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform \
    > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"
git -C "${project_root}" rev-parse HEAD > "${result_dir}/source_commit.txt"
git -C "${project_root}" status --short --branch \
    > "${result_dir}/source_status.txt"

"${project_root}/scripts/run_exp0096.sh" control 1 on on \
    > "${result_dir}/correctness_control_raw.jsonl"
grep '"execution_unit"' "${result_dir}/correctness_control_raw.jsonl" \
    > "${result_dir}/correctness_control.jsonl"
set +e
"${project_root}/scripts/run_exp0096.sh" candidate 1 on on \
    > "${result_dir}/correctness_candidate_raw.jsonl"
candidate_status=$?
set -e
if [[ "${candidate_status}" -eq 0 ]]; then
    printf 'candidate unexpectedly returned success at the failure boundary\n' >&2
    exit 1
fi
grep '"execution_unit"' "${result_dir}/correctness_candidate_raw.jsonl" \
    > "${result_dir}/correctness_candidate.jsonl"

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0096_stage_a.py" \
    "${result_dir}" "${package}" > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/analyze_exp0096_stage_a.py" \
    "${result_dir}" "${package}" --report \
    > "${result_dir}/full_profiling_report.md"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
{
    printf 'experiment=EXP-0096\nsource_head=%s\n' "${source_head}"
    printf 'control=EXP-0084_formal_128kib_2d_lut\n'
    printf 'candidate=old_direct_hvx_fixed_point_swiglu\n'
    printf 'stop_boundary=correctness_gate\n'
    printf 'physical_contract=exact_8mib_vtcm_zero_intermediate_ddr_one_hmx_owner\n'
    printf 'package=%s\nresult_dir=%s\nartifact_dir=%s\n' \
        "${package}" "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
(cd "${artifact_dir}" && \
 sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    > "${result_dir}/artifact_sha256.txt"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
