#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0161/${timestamp}_${source_sha:0:12}_phase_a}"

mkdir -p "${result_dir}/raw"
git -C "${project_root}" status --porcelain > "${result_dir}/source_status.txt"
printf '%s\n' "${source_sha}" > "${result_dir}/source_commit.txt"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_before.txt"

cd "${project_root}"
bash scripts/check_exp0161_static.sh > "${result_dir}/static_gate.log"
bash scripts/build_exp0161.sh > "${result_dir}/build.log"

for length in 64 256 1024 4096; do
    cp "/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/decode_l${length}_w4u8_hmx_delta_v2/manifest.json" \
        "${result_dir}/package_manifest_l${length}.json"
    bash scripts/deploy_exp0161.sh "${length}" \
        > "${result_dir}/deploy_l${length}.log"
done

for repeat_count in 1 10; do
    for round in 1 2 3 4 5; do
        printf -v round_name '%02d' "${round}"
        if (( round % 2 == 1 )); then
            modes=(serial direct pipeline)
        else
            modes=(pipeline direct serial)
        fi
        for length in 64 256 1024 4096; do
            for mode in "${modes[@]}"; do
                output="${result_dir}/raw/l${length}_r${repeat_count}_round_${round_name}_${mode}.jsonl"
                error="${result_dir}/raw/l${length}_r${repeat_count}_round_${round_name}_${mode}.stderr"
                set +e
                bash scripts/run_exp0161.sh \
                    "${length}" "${mode}" "${repeat_count}" on \
                    > "${output}" 2> "${error}"
                status=$?
                set -e
                printf '%s\n' "${status}" > "${output}.status"
            done
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
"${python_exe}" scripts/summarize_exp0161_phase_a.py \
    --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
