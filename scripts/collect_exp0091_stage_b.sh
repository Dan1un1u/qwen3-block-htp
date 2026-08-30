#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0091_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0091_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0091}"
artifact_root="${QBH_EXP0091_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0091/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${result_root}/stage_b_${source_short}"
artifact_dir="${artifact_root}/${source_short}/stage_b"
rounds=5

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0091 Stage B\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
mkdir -p "${result_dir}" "${artifact_dir}"
"${project_root}/scripts/build_exp0091.sh" > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0091_static.sh" \
    > "${result_dir}/static_gate.json"
/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump \
    -d --disassemble-symbols=qbh_unpack_w4_to_s8_hvx,qbh_unpack_w4_to_s8_hvx_interleaved2 \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    > "${result_dir}/decoder.disasm"
QBH_EXP0091_PACKAGE="${package}" \
    "${project_root}/scripts/deploy_exp0091_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_before.txt"
git -C "${project_root}" rev-parse HEAD > "${result_dir}/source_commit.txt"
git -C "${project_root}" status --short --branch > "${result_dir}/source_status.txt"

for mode in control candidate; do
    raw="${result_dir}/correctness_${mode}_raw.jsonl"
    "${project_root}/scripts/run_exp0091.sh" "${mode}" 1 on on > "${raw}"
    grep '"execution_unit"' "${raw}" > "${result_dir}/correctness_${mode}.jsonl"
done
for repeat in 1 10; do
    for mode in control candidate; do
        : > "${result_dir}/paired_${mode}_r${repeat}.jsonl"
    done
done
for ((round = 1; round <= rounds; ++round)); do
    printf 'EXP-0091 Stage-B round %d/%d\n' "${round}" "${rounds}"
    if ((round % 2 == 1)); then
        repeats=(1 10)
        modes=(control candidate)
    else
        repeats=(10 1)
        modes=(candidate control)
    fi
    for repeat in "${repeats[@]}"; do
        for mode in "${modes[@]}"; do
            raw="${result_dir}/round_${round}_${mode}_r${repeat}_raw.jsonl"
            "${project_root}/scripts/run_exp0091.sh" \
                "${mode}" "${repeat}" on off > "${raw}"
            grep '"execution_unit"' "${raw}" \
                >> "${result_dir}/paired_${mode}_r${repeat}.jsonl"
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0091.py" "${result_dir}" \
    > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/analyze_exp0091.py" "${result_dir}" --report \
    > "${result_dir}/full_profiling_report.md"
cp "${package}/manifest.json" "${result_dir}/package_manifest.json"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${artifact_dir}/"
{
    printf 'experiment=EXP-0091\nsource_head=%s\n' "${source_head}"
    printf 'control=exp0084_gate_up_single_vector_vlut32\n'
    printf 'candidate=gate_up_two_way_interleaved_vlut32\n'
    printf 'repeat_contract=repeat1,repeat10\npaired_rounds=%d\n' "${rounds}"
    printf 'physical_contract=exact_8mib_vtcm_zero_intermediate_ddr_one_hmx_owner\n'
    printf 'package=%s\nresult_dir=%s\nartifact_dir=%s\n' \
        "${package}" "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
(cd "${artifact_dir}" && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    > "${result_dir}/artifact_sha256.txt"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
