#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_root="${QBH_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0021}"
artifact_root="${QBH_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0021}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}"
artifact_dir="${artifact_root}/${source_short}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi

python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}/static" "${artifact_dir}"

"${project_root}/scripts/build_exp0021.sh" > "${result_dir}/build.log" 2>&1
QBH_STATIC_OUTPUT_DIR="${result_dir}/static" \
    "${project_root}/scripts/check_exp0021_static.sh" \
    > "${result_dir}/static_gate.json"

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"

"${project_root}/scripts/run_exp0021.sh" 1 1 \
    > "${result_dir}/correctness_repeat1.jsonl"

: > "${result_dir}/timing_repeat1.jsonl"
: > "${result_dir}/timing_repeat10.jsonl"
for round in 1 2 3 4 5 6 7 8 9 10; do
    QBH_SKIP_DEPLOY=1 "${project_root}/scripts/run_exp0021.sh" 1 0 \
        >> "${result_dir}/timing_repeat1.jsonl"
    QBH_SKIP_DEPLOY=1 "${project_root}/scripts/run_exp0021.sh" 10 0 \
        >> "${result_dir}/timing_repeat10.jsonl"
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/validate_exp0021_evidence.py" \
    "${result_dir}" > "${result_dir}/matrix_gate.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_mlp_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"

{
    printf 'experiment=EXP-0021\n'
    printf 'source_branch=%s\n' "$(git -C "${project_root}" branch --show-current)"
    printf 'source_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_middle_block_mlp\n'
    printf 'physical_contract=paired_gate_up_tile_to_vtcm_middle_to_down\n'
    printf 'intermediate_ddr_allowed=false\n'
    printf 'artifact_dir=%s\n' "${artifact_dir}"
} > "${result_dir}/manifest.txt"

(cd "${artifact_dir}" && sha256sum qwen3_mlp_cli libqwen3_probe.so \
    libqwen3_probe_skel.so) > "${result_dir}/artifact_sha256.txt"
sha256sum "${result_dir}"/*.jsonl "${result_dir}"/*.json \
    "${result_dir}/static"/*.txt > "${result_dir}/evidence_sha256.txt"

printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
