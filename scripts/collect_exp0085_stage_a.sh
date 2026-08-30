#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0085_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0085}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_stage_a"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before Stage-A evidence\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"

mkdir -p "${result_dir}"
"${project_root}/scripts/build_exp0085.sh" \
    >"${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0085_stage_a_static.sh" \
    >"${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0085_block.sh" \
    >"${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l >"${result_dir}/adb_devices.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_before.txt"

for variant in F16F16 W4F16 W4U8; do
    stem="${variant,,}"
    "${project_root}/scripts/run_exp0085_stage_a.sh" \
        "${variant}" 1 "${result_dir}/${stem}_timeline.json" \
        >"${result_dir}/${stem}.jsonl"
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" \
    "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0085_stage_a.py" \
    "${result_dir}" >"${result_dir}/stage_a_summary.json"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum >"${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nSOURCE_HEAD=%s\n' "${result_dir}" "${source_head}"
