#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0162/${timestamp}_${source_sha:0:12}_formal}"
control_package="${QBH_EXP0162_CONTROL_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0162/control_delta_capacity104}"
candidate_package="${QBH_EXP0162_CANDIDATE_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0162/candidate_segmented_capacity104}"

mkdir -p "${result_dir}/raw"
git -C "${project_root}" status --porcelain > "${result_dir}/source_status.txt"
test ! -s "${result_dir}/source_status.txt"
printf '%s\n' "${source_sha}" > "${result_dir}/source_commit.txt"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_before.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_before.txt"

cd "${project_root}"
bash scripts/check_exp0162_static.sh > "${result_dir}/static_gate.log"
bash scripts/build_exp0162.sh > "${result_dir}/build.log"
cp "${control_package}/manifest.json" "${result_dir}/control_manifest.json"
cp "${candidate_package}/manifest.json" "${result_dir}/candidate_manifest.json"

bash scripts/deploy_exp0162.sh control > "${result_dir}/deploy_control.log"
bash scripts/deploy_exp0162.sh candidate > "${result_dir}/deploy_candidate.log"
bash scripts/run_exp0162.sh control layout > "${result_dir}/layout_control.jsonl"
bash scripts/run_exp0162.sh candidate layout > "${result_dir}/layout_candidate.jsonl"
bash scripts/run_exp0162.sh control map > "${result_dir}/map_control.jsonl"
bash scripts/run_exp0162.sh candidate map > "${result_dir}/map_candidate.jsonl"

for round in $(seq 1 10); do
    printf -v round_name '%02d' "${round}"
    if (( round % 2 == 1 )); then
        modes=(control candidate)
    else
        modes=(candidate control)
    fi
    for mode in "${modes[@]}"; do
        bash scripts/run_exp0162.sh "${mode}" replay \
            > "${result_dir}/raw/round_${round_name}_${mode}.jsonl" \
            2> "${result_dir}/raw/round_${round_name}_${mode}.stderr"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

"${python_exe}" scripts/summarize_exp0162.py --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"

