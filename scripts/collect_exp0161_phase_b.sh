#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_sha="$(git -C "${project_root}" rev-parse HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0161/${timestamp}_${source_sha:0:12}_phase_b}"

mkdir -p "${result_dir}/raw" "${result_dir}/capture"
git -C "${project_root}" status --porcelain > "${result_dir}/source_status.txt"
printf '%s\n' "${source_sha}" > "${result_dir}/source_commit.txt"
"${adb_exe}" devices -l > "${result_dir}/adb_devices.txt"
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_before.txt"

cd "${project_root}"
bash scripts/check_exp0161_static.sh > "${result_dir}/static_gate.log"
bash scripts/build_exp0161.sh > "${result_dir}/build.log"

# The inherited CLI exit gate still compares scan runs against the pre-EXP-0161
# device golden and cache carrier. EXP-0161 deliberately changed both
# authorities: the independent CPU logical output and logical current-token K/V
# are checked from the captures below. Accept a timing invocation here only
# when the DSP/physical contract itself completed; retain the raw CLI status for
# audit instead of pretending that the legacy self-check passed.
validate_exp0161_run() {
    local jsonl="$1"
    local expected_repeats="$2"
    "${python_exe}" - "${jsonl}" "${expected_repeats}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_repeats = int(sys.argv[2])
record = None
if path.is_file():
    for line in reversed(path.read_text(errors="ignore").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("experiment") == "EXP-0161":
            record = candidate
            break
if record is None:
    raise SystemExit(1)
valid = (
    int(record.get("rpc_result", -1)) == 0
    and int(record.get("dsp_status", -1)) == 3
    and int(record.get("vtcm_requested_bytes", 0)) == 8 * 1024 * 1024
    and int(record.get("vtcm_acquired_bytes", 0)) == 8 * 1024 * 1024
    and int(record.get("block_invocation_count", 0)) == expected_repeats
    and int(record.get("intermediate_ddr_read_bytes", -1)) == 0
    and int(record.get("intermediate_ddr_write_bytes", -1)) == 0
    and int(record.get("intermediate_spill_fill_count", -1)) == 0
    and int(record.get("ledger_unattributed_ticks", -1)) == 0
    and record.get("intermediate_residency") == "VTCM"
)
raise SystemExit(0 if valid else 1)
PY
}

for length in 64 256 1024 4096; do
    cp "/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/decode_l${length}_w4u8_hmx_delta_v2/manifest.json" \
        "${result_dir}/control_manifest_l${length}.json"
    cp "/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/decode_l${length}_w4u8_hmx_segmented_v4b/manifest.json" \
        "${result_dir}/candidate_manifest_l${length}.json"
    bash scripts/deploy_exp0161.sh "${length}" delta \
        > "${result_dir}/deploy_control_l${length}.log"
    bash scripts/deploy_exp0161.sh "${length}" segmented \
        > "${result_dir}/deploy_candidate_l${length}.log"
done

for repeat_count in 1 10; do
    for round in 1 2 3 4 5; do
        printf -v round_name '%02d' "${round}"
        for length in 64 256 1024 4096; do
            control_mode=pipeline
            if (( length == 4096 )); then
                control_mode=serial
            fi
            if (( round % 2 == 1 )); then
                modes=(control candidate)
            else
                modes=(candidate control)
            fi
            for mode in "${modes[@]}"; do
                if [[ "${mode}" == control ]]; then
                    reconstruction="${control_mode}"
                else
                    reconstruction=segmented
                fi
                output="${result_dir}/raw/l${length}_r${repeat_count}_round_${round_name}_${mode}.jsonl"
                error="${result_dir}/raw/l${length}_r${repeat_count}_round_${round_name}_${mode}.stderr"
                status=1
                runner_status=1
                for attempt in 1 2 3; do
                    set +e
                    bash scripts/run_exp0161.sh \
                        "${length}" "${reconstruction}" "${repeat_count}" on \
                        > "${output}" 2> "${error}"
                    runner_status=$?
                    set -e
                    if validate_exp0161_run "${output}" "${repeat_count}"; then
                        status=0
                        break
                    fi
                    sleep 1
                done
                printf '%s\n' "${runner_status}" > "${output}.runner_status"
                printf '%s\n' "${status}" > "${output}.status"
            done
        done
    done
done

for length in 64 256 1024 4096; do
    control_mode=pipeline
    if (( length == 4096 )); then
        control_mode=serial
    fi
    for mode in control candidate; do
        remote_suffix=delta
        reconstruction="${control_mode}"
        if [[ "${mode}" == candidate ]]; then
            remote_suffix=segmented
            reconstruction=segmented
        fi
        remote_root="/data/local/tmp/qwen3-block-htp/exp0161-l${length}-${remote_suffix}"
        remote_capture="${remote_root}/phase_b_capture"
        local_capture="${result_dir}/capture/l${length}_${mode}"
        mkdir -p "${local_capture}"
        "${adb_exe}" shell \
            "mkdir -p ${remote_capture} && rm -f ${remote_capture}/block_output_u8.bin ${remote_capture}/actual_kv_cache_k_u8.bin ${remote_capture}/actual_kv_cache_v_u8.bin"
        capture_status=1
        capture_runner_status=1
        for attempt in 1 2 3; do
            set +e
            QBH_EXP0161_DUMP_CACHE_REMOTE="${remote_capture}" \
            QBH_EXP0161_DUMP_OUTPUT_REMOTE="${remote_capture}/block_output_u8.bin" \
                bash scripts/run_exp0161.sh \
                    "${length}" "${reconstruction}" 1 on \
                    > "${local_capture}/run.jsonl" \
                    2> "${local_capture}/run.stderr"
            capture_runner_status=$?
            set -e
            if validate_exp0161_run "${local_capture}/run.jsonl" 1; then
                capture_status=0
                break
            fi
            sleep 1
        done
        printf '%s\n' "${capture_runner_status}" \
            > "${local_capture}/run.jsonl.runner_status"
        printf '%s\n' "${capture_status}" \
            > "${local_capture}/run.jsonl.status"
        if (( capture_status != 0 )); then
            printf 'capture failed for L%s %s\n' \
                "${length}" "${mode}" >&2
            exit "${capture_status}"
        fi
        for file in block_output_u8.bin actual_kv_cache_k_u8.bin actual_kv_cache_v_u8.bin; do
            "${adb_exe}" pull "${remote_capture}/${file}" \
                "$(wslpath -w "${local_capture}/${file}")" >/dev/null
        done
    done
done

"${adb_exe}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r' \
    > "${result_dir}/boot_id_after.txt"
cmp -s "${result_dir}/boot_id_before.txt" "${result_dir}/boot_id_after.txt"
"${python_exe}" scripts/summarize_exp0161_phase_b.py \
    --result-dir "${result_dir}"
find "${result_dir}" -type f ! -name evidence_sha256.txt -print0 | \
    sort -z | xargs -0 sha256sum > "${result_dir}/evidence_sha256.txt"
printf 'RESULT_DIR=%s\n' "${result_dir}"
