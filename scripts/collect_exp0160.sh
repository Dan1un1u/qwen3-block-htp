#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0160/${timestamp}_${source_sha:0:12}_formal}"
package="${QBH_EXP0159_CANDIDATE_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0159/w4u8_hmx_delta_v2_compact}"

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
bash scripts/check_exp0160_static.sh > "${result_dir}/static_gate.log"
bash scripts/build_exp0160.sh > "${result_dir}/build.log"
test -f "${package}/manifest.json"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
QBH_DEPLOY_BINARIES_ONLY=0 bash scripts/deploy_exp0159.sh candidate \
    > "${result_dir}/deploy.log"
QBH_W4U8_DELTA_RECONSTRUCTION=pipeline \
    bash scripts/run_exp0160.sh candidate layout \
    > "${result_dir}/layout_pipeline.jsonl"

for round in $(seq 1 10); do
    printf -v round_name '%02d' "${round}"
    if (( round % 2 == 1 )); then
        modes=(serial pipeline)
    else
        modes=(pipeline serial)
    fi
    for mode in "${modes[@]}"; do
        QBH_W4U8_DELTA_RECONSTRUCTION="${mode}" \
            bash scripts/run_exp0160.sh candidate replay \
            > "${result_dir}/raw/round_${round_name}_${mode}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

"${python_exe}" scripts/summarize_exp0160.py --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
