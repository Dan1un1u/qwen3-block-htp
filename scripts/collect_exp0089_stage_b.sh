#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0089_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0089}"
artifact_root="${QBH_EXP0089_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0089/artifacts}"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_b_${source_short}"
artifact_dir="${artifact_root}/${source_short}/stage_b"
rounds=5

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0089 Stage B\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"
"${project_root}/scripts/check_exp0089_stage_b_static.sh" \
    > "${result_dir}/static_gate.json"
"${project_root}/scripts/build_exp0089.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/deploy_exp0089.sh" > "${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
git -C "${project_root}" rev-parse HEAD > "${result_dir}/source_commit.txt"
git -C "${project_root}" status --short --branch > "${result_dir}/source_status.txt"

for mode in control candidate; do
    prestage=0
    [[ "${mode}" == candidate ]] && prestage=2
    raw="${result_dir}/correctness_${mode}_raw.jsonl"
    "${project_root}/scripts/run_exp0089.sh" 1 on on 1 "${prestage}" \
        > "${raw}"
    grep '"execution_unit"' "${raw}" \
        > "${result_dir}/correctness_${mode}.jsonl"
    grep '"record":"gate_up_down_audit"' "${raw}" \
        > "${result_dir}/correctness_${mode}_boundary.jsonl"
    if [[ "${mode}" == candidate ]]; then
        grep '"record":"gate_up_down_prestage"' "${raw}" \
            > "${result_dir}/correctness_candidate_prestage.jsonl"
    fi
done

for repeat in 1 10; do
    for mode in control candidate; do
        : > "${result_dir}/paired_${mode}_r${repeat}.jsonl"
        : > "${result_dir}/paired_${mode}_r${repeat}_raw.jsonl"
    done
    : > "${result_dir}/paired_candidate_r${repeat}_prestage.jsonl"
done

for ((round = 1; round <= rounds; ++round)); do
    printf 'EXP-0089 Stage-B round %d/%d\n' "${round}" "${rounds}"
    if ((round % 2 == 1)); then
        repeats=(1 10)
        modes=(control candidate)
    else
        repeats=(10 1)
        modes=(candidate control)
    fi
    for repeat in "${repeats[@]}"; do
        for mode in "${modes[@]}"; do
            prestage=0
            [[ "${mode}" == candidate ]] && prestage=2
            run_raw="${result_dir}/round_${round}_${mode}_r${repeat}_raw.jsonl"
            "${project_root}/scripts/run_exp0089.sh" \
                "${repeat}" on off 0 "${prestage}" > "${run_raw}"
            cat "${run_raw}" \
                >> "${result_dir}/paired_${mode}_r${repeat}_raw.jsonl"
            grep '"execution_unit"' "${run_raw}" \
                >> "${result_dir}/paired_${mode}_r${repeat}.jsonl"
            if [[ "${mode}" == candidate ]]; then
                grep '"record":"gate_up_down_prestage"' "${run_raw}" \
                    >> "${result_dir}/paired_candidate_r${repeat}_prestage.jsonl"
            fi
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0089_stage_b.py" \
    "${result_dir}" > "${result_dir}/gate_summary.json"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
sha256sum "${artifact_dir}"/* > "${artifact_dir}/sha256sums.txt"
sha256sum "${result_dir}/gate_summary.json" \
    > "${result_dir}/gate_summary.sha256"
printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'ARTIFACT_DIR=%s\n' "${artifact_dir}"
