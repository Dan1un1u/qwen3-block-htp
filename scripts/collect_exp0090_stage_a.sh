#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0090_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0090}"
artifact_root="${QBH_EXP0090_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0090/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_a_${source_short}"
artifact_dir="${artifact_root}/${source_short}/stage_a"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before evidence collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"

mkdir -p "${result_dir}/audit_runs" "${artifact_dir}"
"${project_root}/scripts/build_exp0090.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0090_stage_a_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0090_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

"${project_root}/scripts/run_exp0090.sh" 0 1 on off \
    > "${result_dir}/ordinary_control.jsonl"
"${project_root}/scripts/run_exp0090.sh" 1 1 on on \
    > "${result_dir}/correctness_audit.jsonl"
for round in $(seq 1 7); do
    "${project_root}/scripts/run_exp0090.sh" 1 1 on off \
        > "${result_dir}/audit_runs/round_${round}.jsonl"
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0090_stage_a.py" "${result_dir}" \
    > "${result_dir}/stage_a_summary.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
{
    printf 'experiment=EXP-0090\nstage=A\nsource_head=%s\n' "${source_head}"
    printf 'control=EXP-0084 W4U8 single FIFO\n'
    printf 'candidate=audit-only queue telemetry\n'
    printf 'result_dir=%s\nartifact_dir=%s\n' "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
find "${result_dir}" -type f ! -name evidence.sha256 -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence.sha256"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
