#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0158/${timestamp}_${source_sha:0:12}_formal}"
f16_package="${QBH_EXP0158_F16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0158/f16f16}"
w4_package="${QBH_EXP0158_W4_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0158/w4f16}"

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
./scripts/check_exp0158_static.sh > "${result_dir}/static_gate.log"
./scripts/build_exp0158.sh > "${result_dir}/build.log"
"${python_exe}" ./scripts/prepare_exp0158_f16_cache_packages.py \
    "${f16_package}" "${w4_package}" > "${result_dir}/package_prepare.log"
cp "${f16_package}/manifest.json" "${result_dir}/f16f16_package_manifest.json"
cp "${w4_package}/manifest.json" "${result_dir}/w4f16_package_manifest.json"
./scripts/deploy_exp0158.sh f16f16 > "${result_dir}/deploy_f16f16.log"
./scripts/deploy_exp0158.sh w4f16 > "${result_dir}/deploy_w4f16.log"

for recipe in f16f16 w4f16; do
    for layout in row_major hmx_native_f16; do
        ./scripts/run_exp0158.sh "${recipe}" "${layout}" layout \
            > "${result_dir}/layout_${recipe}_${layout}.jsonl"
    done
done

variants=(
    f16f16_row_major
    f16f16_hmx_native_f16
    w4f16_row_major
    w4f16_hmx_native_f16
)
for round in $(seq 1 10); do
    printf -v round_name '%02d' "${round}"
    offset=$(( (round - 1) % 4 ))
    for position in 0 1 2 3; do
        key="${variants[$(( (position + offset) % 4 ))]}"
        recipe="${key%%_*}"
        layout="${key#${recipe}_}"
        ./scripts/run_exp0158.sh "${recipe}" "${layout}" replay \
            > "${result_dir}/raw/round_${round_name}_${key}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery > "${result_dir}/battery_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

"${python_exe}" ./scripts/summarize_exp0158.py --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
