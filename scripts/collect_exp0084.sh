#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_package="${QBH_EXP0084_FP16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
u8_package="${QBH_EXP0084_U8_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
result_root="${QBH_EXP0084_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0084}"
artifact_root="${QBH_EXP0084_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0084/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
source_branch="$(git -C "${project_root}" branch --show-current)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_formal"
artifact_dir="${artifact_root}/${source_short}/formal"
paired_rounds=7
grid_rounds=2

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before formal evidence collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"

mkdir -p "${result_dir}" "${artifact_dir}" \
    "${result_dir}/correctness" "${result_dir}/stages" \
    "${result_dir}/tri_variant"
"${project_root}/scripts/build_exp0084.sh" \
    > "${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0084_static.sh" \
    > "${result_dir}/static_gate.json"
QBH_EXP0084_FP16_PACKAGE="${fp16_package}" \
QBH_EXP0084_U8_PACKAGE="${u8_package}" \
    "${project_root}/scripts/deploy_exp0084_block.sh" \
    > "${result_dir}/deploy.log" 2>&1

"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model \
    > "${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform \
    > "${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    > "${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_before.txt"

printf 'EXP-0084 equal-budget FP16 grid\n'
"${project_root}/scripts/search_exp0084_fp16_norm_grid.sh" \
    "${result_dir}/grid" "${grid_rounds}" \
    > "${result_dir}/grid.log"
python3 "${project_root}/scripts/select_exp0084_grid.py" \
    "${result_dir}/grid" > "${result_dir}/grid_selection.json"
read -r selected_rows selected_contexts < <(
    python3 "${project_root}/scripts/select_exp0084_grid.py" \
        "${result_dir}/grid" --plain
)
read -r f16_best_rows f16_best_contexts \
        w4_best_rows w4_best_contexts < <(
    python3 - "${result_dir}/grid_selection.json" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1]))["recipe_specific_best"]
print(
    selection["f16f16"]["rows_per_task"],
    selection["f16f16"]["contexts"],
    selection["w4f16"]["rows_per_task"],
    selection["w4f16"]["contexts"],
)
PY
)
export EXP0084_FP16_NORM_ROWS="${selected_rows}"
export EXP0084_FP16_NORM_CONTEXTS="${selected_contexts}"

printf 'EXP-0084 correctness matrix (rows=%s contexts=%s)\n' \
    "${selected_rows}" "${selected_contexts}"
for variant in F16F16 W4F16; do
    for mode in control stage_a stage_b stage_c candidate; do
        "${project_root}/scripts/run_exp0084.sh" \
            "${variant}" "${mode}" 1 on on \
            > "${result_dir}/correctness/${variant,,}_${mode}.jsonl"
    done
done
"${project_root}/scripts/run_exp0084.sh" W4U8 control 1 on on \
    > "${result_dir}/correctness/w4u8_control.jsonl"

for stage in stage_a stage_b stage_c candidate; do
    printf 'EXP-0084 %s paired collection (%d rounds)\n' \
        "${stage}" "${paired_rounds}"
    "${project_root}/scripts/run_exp0084_stage_smoke.sh" \
        "${stage}" "${result_dir}/stages/${stage}" "${paired_rounds}"
done

printf 'EXP-0084 interleaved canonical and recipe-best tri-variant formal\n'
"${project_root}/scripts/run_exp0084_tri_formal.sh" \
    "${result_dir}/tri_variant" \
    "${selected_rows}" "${selected_contexts}" \
    "${f16_best_rows}" "${f16_best_contexts}" \
    "${w4_best_rows}" "${w4_best_contexts}" \
    "${paired_rounds}"

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    > "${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" \
    "${result_dir}/boot_id_after.txt"

cp "${fp16_package}/manifest.json" \
    "${result_dir}/fp16_package_manifest.json"
cp "${u8_package}/manifest.json" \
    "${result_dir}/w4u8_package_manifest.json"
python3 "${project_root}/scripts/validate_exp0084.py" "${result_dir}" \
    > "${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/validate_exp0084.py" \
    "${result_dir}" --report \
    > "${result_dir}/full_profiling_report.md"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
{
    printf 'experiment=EXP-0084\nsource_branch=%s\nsource_head=%s\n' \
        "${source_branch}" "${source_head}"
    printf 'execution_unit=qwen3_layer14_complete_block_m64\n'
    printf 'fp16_control=EXP-0038-compatible-common-source-control\n'
    printf 'w4u8_control=EXP-0079-byte-identical-schedule\n'
    printf 'candidate=input_norm_pool_post_norm_pool\n'
    printf 'rejected_stage_a=qk_norm_rope_head_pairs\n'
    printf 'repeat_contract=repeat1,repeat10\npaired_rounds=%d\n' \
        "${paired_rounds}"
    printf 'grid_rounds=%d\nselected_rows_per_task=%s\nselected_contexts=%s\n' \
        "${grid_rounds}" "${selected_rows}" "${selected_contexts}"
    printf 'f16f16_best_rows_per_task=%s\nf16f16_best_contexts=%s\n' \
        "${f16_best_rows}" "${f16_best_contexts}"
    printf 'w4f16_best_rows_per_task=%s\nw4f16_best_contexts=%s\n' \
        "${w4_best_rows}" "${w4_best_contexts}"
    printf 'physical_contract=exact_8mib_vtcm_one_fastrpc_one_hmx_owner\n'
    printf 'intermediate_ddr_allowed=false\nqnn_allowed=false\n'
    printf 'fp16_package=%s\nw4u8_package=%s\n' \
        "${fp16_package}" "${u8_package}"
    printf 'result_dir=%s\nartifact_dir=%s\n' \
        "${result_dir}" "${artifact_dir}"
} > "${result_dir}/manifest.txt"
(cd "${artifact_dir}" && sha256sum \
    qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    > "${result_dir}/artifact_sha256.txt"
find "${fp16_package}" "${u8_package}" -maxdepth 1 -type f \
    -print0 | sort -z | xargs -0 sha256sum \
    > "${result_dir}/package_sha256.txt"
find "${result_dir}" -type f \
    ! -name evidence_sha256.txt -print0 | sort -z | xargs -0 sha256sum \
    > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' \
    "${result_dir}" "${artifact_dir}"
