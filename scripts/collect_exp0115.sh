#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_package="${QBH_EXP0115_FP16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
result_root="${QBH_EXP0115_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0115}"
artifact_root="${QBH_EXP0115_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0115/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_formal"
artifact_dir="${artifact_root}/${source_short}/${timestamp}_formal"
cells=(control candidate)

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0115 formal collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"

"${project_root}/scripts/build_exp0115.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0115_static.sh" > "${result_dir}/static_gate.json"
QBH_EXP0115_FP16_PACKAGE="${fp16_package}" \
    "${project_root}/scripts/deploy_exp0115_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
git -C "${project_root}" rev-parse HEAD > "${result_dir}/source_commit.txt"
git -C "${project_root}" status --short --branch > "${result_dir}/source_status.txt"

for cell in "${cells[@]}"; do
    raw="${result_dir}/correctness_${cell}_raw.jsonl"
    "${project_root}/scripts/run_exp0115.sh" "${cell}" 1 on on > "${raw}"
    grep execution_unit "${raw}" > "${result_dir}/correctness_${cell}.jsonl"
    : > "${result_dir}/paired_${cell}_r1.jsonl"
    : > "${result_dir}/paired_${cell}_r10.jsonl"
done

for round in 1 2 3 4 5; do
    printf 'EXP-0115 round %d/5\n' "${round}"
    if ((round % 2 == 1)); then
        repeats=(1 10)
        order=(control candidate)
    else
        repeats=(10 1)
        order=(candidate control)
    fi
    for repeat in "${repeats[@]}"; do
        for cell in "${order[@]}"; do
            raw="${result_dir}/round_${round}_${cell}_r${repeat}_raw.jsonl"
            "${project_root}/scripts/run_exp0115.sh" \
                "${cell}" "${repeat}" on off > "${raw}"
            grep execution_unit "${raw}" \
                >> "${result_dir}/paired_${cell}_r${repeat}.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0115.py" "${result_dir}" \
    > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/analyze_exp0115.py" "${result_dir}" --report \
    > "${result_dir}/full_profiling_report.md"

cp "${fp16_package}/manifest.json" "${result_dir}/fp16_package_manifest.json"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"
{
    printf 'experiment=EXP-0115\nsource_head=%s\n' "${source_head}"
    printf 'parent=EXP-0112\ncells=%s\n' "${cells[*]}"
    printf 'recipe=W4F16\nvariable=gate_up_low_nibble_vlut16_mask\n'
    printf 'repeat_contract=repeat1,repeat10\nrounds=5\n'
    printf 'physical_contract=exact_8mib_vtcm_zero_intermediate_ddr_one_hmx_owner\n'
    printf 'result_dir=%s\nartifact_dir=%s\n' "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
(cd "${artifact_dir}" && \
    sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    > "${result_dir}/artifact_sha256.txt"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
