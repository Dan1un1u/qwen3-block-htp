#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0157/${timestamp}_${source_sha:0:12}_formal}"
package="${QBH_EXP0157_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0156/w4u8_hmx_native_v1}"
remote_root="${EXP0157_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0157-prefill-carrier-reuse}"

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
./scripts/check_exp0157_static.sh > "${result_dir}/static_gate.log"
./scripts/build_exp0157.sh > "${result_dir}/build.log"

QBH_EXP0157_DEPLOY=1 EXP0157_REMOTE_ROOT="${remote_root}" \
    ./scripts/run_exp0157.sh reuse layout \
    > "${result_dir}/deploy_and_layout_reuse.log"
EXP0157_REMOTE_ROOT="${remote_root}" \
    ./scripts/run_exp0157.sh duplicate layout \
    > "${result_dir}/layout_duplicate.log"

for round in $(seq 1 10); do
    if (( round % 2 == 1 )); then
        modes=(duplicate reuse)
    else
        modes=(reuse duplicate)
    fi
    for mode in "${modes[@]}"; do
        printf -v round_name '%02d' "${round}"
        EXP0157_REMOTE_ROOT="${remote_root}" \
            ./scripts/run_exp0157.sh "${mode}" replay \
            > "${result_dir}/raw/round_${round_name}_${mode}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

/home/daniuniu/.cache/qwen3-block-htp-py/bin/python \
    ./scripts/summarize_exp0157.py --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
