#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python3}"
model_root="${QBH_EXP0149_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149}"
result_root="${QBH_EXP0149_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0149}"
artifact_root="${QBH_EXP0149_ARTIFACT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149/artifacts}"
source_head="$(git -C "${project_root}" rev-parse HEAD)"
source_short="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="${result_root}/${timestamp}_${source_short}_formal"
artifact_dir="${artifact_root}/${source_short}/${timestamp}_formal"
recipes=(f16f16 w4f16 w4u8)
declare -A packages=(
    [f16f16]="${model_root}/f16f16_formal"
    [w4f16]="${model_root}/w4f16_formal"
    [w4u8]="${model_root}/w4u8_formal"
)

if ! git -C "${project_root}" diff --quiet || \
   ! git -C "${project_root}" diff --cached --quiet || \
   [[ -n "$(git -C "${project_root}" ls-files --others --exclude-standard)" ]]; then
    printf 'source worktree must be clean before EXP-0149 formal collection\n' >&2
    exit 1
fi
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
    preflight --source-worktree "${project_root}"
for recipe in "${recipes[@]}"; do
    test -f "${packages[${recipe}]}/manifest.json"
done

mkdir -p "${result_dir}/raw" "${result_dir}/packages" "${artifact_dir}"
"${project_root}/scripts/build_exp0149.sh" >"${result_dir}/build.log" 2>&1
"${project_root}/scripts/check_exp0149_static.sh" \
    >"${result_dir}/static_gate.json"

"${adb_exe}" devices -l >"${result_dir}/adb_devices.txt"
"${adb_exe}" shell getprop ro.product.model >"${result_dir}/device_model.txt"
"${adb_exe}" shell getprop ro.board.platform >"${result_dir}/device_platform.txt"
"${adb_exe}" shell getprop ro.build.fingerprint \
    >"${result_dir}/device_fingerprint.txt"
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_before.txt"
"${adb_exe}" shell dumpsys battery >"${result_dir}/battery_before.txt"
git -C "${project_root}" rev-parse HEAD >"${result_dir}/source_commit.txt"
git -C "${project_root}" status --short --branch \
    >"${result_dir}/source_status.txt"

for recipe in "${recipes[@]}"; do
    remote_root="/data/local/tmp/qwen3-block-htp/exp0149-${recipe}"
    EXP0147_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0147_package.sh" \
        "${packages[${recipe}]}" \
        >"${result_dir}/deploy_${recipe}.log" 2>&1
    cp "${packages[${recipe}]}/manifest.json" \
        "${result_dir}/packages/${recipe}_manifest.json"
    (cd "${packages[${recipe}]}" && sha256sum manifest.json) \
        >"${result_dir}/packages/${recipe}_manifest_sha256.txt"
done

# One capture sequence per recipe supplies the formal numerical evidence.
# EXP-0149 deliberately keeps the DSP's single-block internal audit disabled:
# the three-layer slice rebinds each layer's buffers and that legacy telemetry
# contract is not slice-safe. Host output metrics plus the captured final
# outputs and physical cache carriers remain available. F16F16 and W4F16 are
# checked offline against the untouched independent software package with the
# approved composition-aware tolerance. Timing evidence below also disables
# the expensive audit and starts from a fresh Prepared Decode Session each run.
mkdir -p "${result_dir}/independent"
for recipe in "${recipes[@]}"; do
    QBH_EXP0149_PACKAGE="${packages[${recipe}]}" \
        QBH_EXP0149_NUMERICAL_AUDIT=off \
        QBH_REPLAY_CAPTURE_ROOT="${result_dir}/independent/capture_${recipe}" \
        "${project_root}/scripts/run_exp0149_replay.sh" "${recipe}" \
        >"${result_dir}/raw/correctness_${recipe}.jsonl"
done
for recipe in f16f16 w4f16; do
    "${python_exe}" \
        "${project_root}/scripts/audit_exp0149_fp16_composition_tolerance.py" \
        --recipe "${recipe}" --package-root "${model_root}" \
        --capture-root "${result_dir}/independent" \
        --output "${result_dir}/independent_${recipe}_composition_gate.json" \
        >"${result_dir}/independent_${recipe}_composition_gate.stdout"
done

# Ten round-robin sequences.  The launch order rotates so no recipe always
# receives the same thermal/clock position.  Round 1 is the repeat-one view;
# the median of all ten fresh sequences is the repeat-ten view.
for round in $(seq 1 10); do
    case $(((round - 1) % 3)) in
        0) order=(f16f16 w4f16 w4u8) ;;
        1) order=(w4f16 w4u8 f16f16) ;;
        2) order=(w4u8 f16f16 w4f16) ;;
    esac
    printf 'EXP-0149 formal round %02d/10: %s\n' \
        "${round}" "${order[*]}"
    for recipe in "${order[@]}"; do
        QBH_EXP0149_PACKAGE="${packages[${recipe}]}" \
            QBH_EXP0149_NUMERICAL_AUDIT=off \
            "${project_root}/scripts/run_exp0149_replay.sh" "${recipe}" \
            >"${result_dir}/raw/round_$(printf '%02d' "${round}")_${recipe}.jsonl"
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id \
    >"${result_dir}/boot_id_after.txt"
"${adb_exe}" shell dumpsys battery >"${result_dir}/battery_after.txt"
cmp "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"

python3 "${project_root}/scripts/summarize_exp0149.py" \
    --result-dir "${result_dir}" --artifact-dir "${artifact_dir}" \
    --json "${result_dir}/gate_summary.json" \
    --markdown "${result_dir}/full_profiling_report.md"

cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${artifact_dir}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${artifact_dir}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${artifact_dir}/"
{
    printf 'experiment=EXP-0149\nsource_head=%s\n' "${source_head}"
    printf 'execution_unit=qwen3_layers13_14_15_vertical_slice_prefill_continuous_decode\n'
    printf 'recipes=%s\n' "${recipes[*]}"
    printf 'repeat1=round_01_one_fresh_sequence\n'
    printf 'repeat10=median_of_ten_fresh_continuous_sequences\n'
    printf 'prefill=positions_0_through_63\n'
    printf 'decode=positions_64_through_71_continuous_cache_append\n'
    printf 'fp16_gate=abs_error_le_0p0625_plus_0p002_times_abs_reference_cosine_ge_0p99999_no_nonfinite\n'
    printf 'physical_contract=exact_8mib_vtcm_zero_intermediate_ddr_one_fastrpc_per_step_one_hmx_owner\n'
    printf 'result_dir=%s\nartifact_dir=%s\n' "${result_dir}" "${artifact_dir}"
} >"${result_dir}/manifest.txt"
(cd "${artifact_dir}" && \
    sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so) \
    >"${result_dir}/artifact_sha256.txt"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum >"${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\nARTIFACT_DIR=%s\n' "${result_dir}" "${artifact_dir}"
