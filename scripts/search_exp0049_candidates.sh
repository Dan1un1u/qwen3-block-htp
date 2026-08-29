#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
batch4_root="${BATCH4_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0049-b4-v1}"
batch8_root="${BATCH8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0049-block}"
result_dir="${QBH_EXP0049_SEARCH_RESULT_DIR:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0049/candidate_search_batch4_batch8}"
batch4_skel="${QBH_EXP0049_BATCH4_SKEL:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0049/candidates/batch4/libqwen3_probe_skel.so}"

mkdir -p "${result_dir}"
"${adb_exe}" shell \
    "mkdir -p ${batch4_root} && test -d ${batch8_root}/block_package_layer14_m64 && cp -R ${batch8_root}/block_package_layer14_m64 ${batch4_root}/"
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
    "${batch4_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
    "${batch4_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${batch4_skel}")" \
    "${batch4_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" shell \
    "chmod 755 ${batch4_root}/qwen3_block_cli ${batch4_root}/libqwen3_probe.so ${batch4_root}/libqwen3_probe_skel.so"

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"
for mode in batch4 batch8; do
    for repeat in 1 10; do
        : > "${result_dir}/${mode}_r${repeat}.jsonl"
    done
done

for round in 1 2 3 4 5 6 7; do
    if (( round % 2 == 1 )); then
        modes=(batch4 batch8)
        repeats=(1 10)
    else
        modes=(batch8 batch4)
        repeats=(10 1)
    fi
    for repeat in "${repeats[@]}"; do
        for mode in "${modes[@]}"; do
            if [[ "${mode}" == batch4 ]]; then
                remote_root="${batch4_root}"
            else
                remote_root="${batch8_root}"
            fi
            REMOTE_ROOT="${remote_root}" \
                "${project_root}/scripts/run_exp0048_stage_a.sh" \
                o_native "${repeat}" on off \
                >> "${result_dir}/${mode}_r${repeat}.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" \
    "${result_dir}/boot_id_after.txt"

python3 - "${result_dir}" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
for mode in ("batch4", "batch8"):
    for repeat in (1, 10):
        rows = [
            json.loads(line)
            for line in (root / f"{mode}_r{repeat}.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]

        def median_per_block(key: str) -> float:
            return statistics.median(
                row[key] / row["block_invocation_count"]
                for row in rows
            )

        print(
            mode,
            repeat,
            len(rows),
            "host_ns",
            median_per_block("host_wall_ns"),
            "gate_up_ticks",
            median_per_block("w4u8_mlp_gate_up_pipeline_ticks"),
            "commands",
            median_per_block("hmx_command_count"),
            "expanded_wait",
            median_per_block("w4u8_mlp_expanded_slot_wait_ticks"),
        )
PY
