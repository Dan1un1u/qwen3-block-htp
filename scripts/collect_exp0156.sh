#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0156/${timestamp}_${source_sha:0:12}_formal}"
package="${QBH_EXP0156_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0156/w4u8_hmx_native_v1}"
remote_root="${EXP0156_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0156-full-stack}"

mkdir -p "${result_dir}/raw"
git -C "${project_root}" status --porcelain > "${result_dir}/source_status.txt"
test ! -s "${result_dir}/source_status.txt"
printf '%s\n' "${source_sha}" > "${result_dir}/source_commit.txt"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_before.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_before.txt"

cd "${project_root}"
./scripts/check_exp0156_static.sh > "${result_dir}/static_gate.log"
./scripts/build_exp0156.sh > "${result_dir}/build.log"

QBH_EXP0156_DEPLOY=1 EXP0156_REMOTE_ROOT="${remote_root}" \
    ./scripts/run_exp0156.sh hmx_native_u8 layout \
    > "${result_dir}/deploy_and_layout_hmx_native_u8.log"
EXP0156_REMOTE_ROOT="${remote_root}" \
    ./scripts/run_exp0156.sh row_major layout \
    > "${result_dir}/layout_row_major.log"

for round in $(seq 1 10); do
    if (( round % 2 == 1 )); then
        layouts=(row_major hmx_native_u8)
    else
        layouts=(hmx_native_u8 row_major)
    fi
    for layout in "${layouts[@]}"; do
        printf -v round_name '%02d' "${round}"
        EXP0156_REMOTE_ROOT="${remote_root}" \
            ./scripts/run_exp0156.sh "${layout}" replay \
            > "${result_dir}/raw/round_${round_name}_${layout}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

/home/daniuniu/.cache/qwen3-block-htp-py/bin/python \
    ./scripts/summarize_exp0156.py --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
