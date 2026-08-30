#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:?output directory required}"
rounds="${2:-5}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"

mkdir -p "${output_dir}"
"${project_root}/scripts/check_exp0087_static.sh" \
    > "${output_dir}/static_gate.json"
git -C "${project_root}" rev-parse HEAD > "${output_dir}/source_head.txt"
sha256sum \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    > "${output_dir}/artifact_sha256.txt"
"${adb_exe}" devices -l > "${output_dir}/adb_devices.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_before.txt"

for repeat_count in 1 10; do
    : > "${output_dir}/control_r${repeat_count}.jsonl"
    : > "${output_dir}/candidate_r${repeat_count}.jsonl"
done

for ((round_index = 1; round_index <= rounds; ++round_index)); do
    for repeat_count in 1 10; do
        if (((round_index + repeat_count) % 2 == 0)); then
            plans=(control candidate)
        else
            plans=(candidate control)
        fi
        for plan in "${plans[@]}"; do
            if [[ "${plan}" == control ]]; then
                slices=2
            else
                slices=4
            fi
            "${project_root}/scripts/run_exp0087.sh" \
                "${repeat_count}" on off off "${slices}" \
                >> "${output_dir}/${plan}_r${repeat_count}.jsonl"
        done
    done
done

"${project_root}/scripts/run_exp0087.sh" 1 on on off 2 \
    > "${output_dir}/correctness_control.jsonl"
"${project_root}/scripts/run_exp0087.sh" 1 on on off 4 \
    > "${output_dir}/correctness_candidate.jsonl"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${output_dir}/boot_id_after.txt"
cmp "${output_dir}/boot_id_before.txt" "${output_dir}/boot_id_after.txt"

"${project_root}/scripts/analyze_exp0087_stage_b.py" \
    "${output_dir}" --output "${output_dir}/stage_b_summary.json"
printf 'OUTPUT_DIR=%s\n' "${output_dir}"
