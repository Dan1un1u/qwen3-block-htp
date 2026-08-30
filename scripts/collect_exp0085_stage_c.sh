#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_root="${QBH_EXP0085_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0085}"
artifact_root="${QBH_EXP0085_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0085/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_stage_c_formal"
artifact_dir="${artifact_root}/${source_short}/stage_c_formal"
rounds="${QBH_EXP0085_STAGE_C_ROUNDS:-7}"
candidate="q_prefix4_k_all"

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before Stage-C evidence\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"

mkdir -p "${result_dir}/correctness" "${result_dir}/performance" \
    "${artifact_dir}"
"${project_root}/scripts/build_exp0085.sh" \
    >"${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0085_stage_a_static.sh" \
    >"${result_dir}/static_gate.json"
"${project_root}/scripts/deploy_exp0085_block.sh" \
    >"${result_dir}/deploy.log" 2>&1
"${adb_exe}" devices -l >"${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model >"${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform >"${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    >"${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_before.txt"

for variant in F16F16 W4F16 W4U8; do
    "${project_root}/scripts/run_exp0085.sh" \
        "${variant}" 1 control on on \
        >"${result_dir}/correctness/${variant,,}_control.jsonl"
    "${project_root}/scripts/run_exp0085.sh" \
        "${variant}" 1 "${candidate}" on on \
        >"${result_dir}/correctness/${variant,,}_candidate.jsonl"
done

for variant in f16f16 w4f16 w4u8; do
    for repeat in 1 10; do
        : >"${result_dir}/performance/${variant}_control_r${repeat}.jsonl"
        : >"${result_dir}/performance/${variant}_candidate_r${repeat}.jsonl"
    done
done

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        schedules=(control "${candidate}")
        repeats=(1 10)
    else
        schedules=("${candidate}" control)
        repeats=(10 1)
    fi
    case $(((round - 1) % 3)) in
        0) variants=(F16F16 W4F16 W4U8) ;;
        1) variants=(W4F16 W4U8 F16F16) ;;
        2) variants=(W4U8 F16F16 W4F16) ;;
    esac
    for schedule in "${schedules[@]}"; do
        side="candidate"
        if [[ "${schedule}" == "control" ]]; then
            side="control"
        fi
        for repeat in "${repeats[@]}"; do
            for variant in "${variants[@]}"; do
                "${project_root}/scripts/run_exp0085.sh" \
                    "${variant}" "${repeat}" "${schedule}" on off \
                    >>"${result_dir}/performance/${variant,,}_${side}_r${repeat}.jsonl"
            done
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
python3 "${project_root}/scripts/analyze_exp0085_stage_c.py" \
    "${result_dir}" >"${result_dir}/gate_summary.json"
python3 "${project_root}/scripts/analyze_exp0085_stage_c.py" \
    --report "${result_dir}" >"${result_dir}/full_profiling_report.md"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
{
    printf 'experiment=EXP-0085\nsource_head=%s\n' "${source_head}"
    printf 'candidate=%s\nrounds=%s\nrepeat_contract=1,10\n' \
        "${candidate}" "${rounds}"
    printf 'physical_contract=exact_8mib_vtcm_one_fastrpc_one_hmx_owner_zero_intermediate_ddr\n'
    printf 'result_dir=%s\nartifact_dir=%s\n' \
        "${result_dir}" "${artifact_dir}"
} >"${result_dir}/manifest.txt"
(cd "${artifact_dir}" && sha256sum \
    qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    >"${result_dir}/artifact_sha256.txt"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum >"${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' \
    "${result_dir}" "${artifact_dir}"
