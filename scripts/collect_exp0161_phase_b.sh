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
                for attempt in 1 2 3; do
                    set +e
                    bash scripts/run_exp0161.sh \
                        "${length}" "${reconstruction}" "${repeat_count}" on \
                        > "${output}" 2> "${error}"
                    status=$?
                    set -e
                    if (( status == 0 )); then
                        break
                    fi
                    sleep 1
                done
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
        "${adb_exe}" shell "mkdir -p ${remote_capture}"
        capture_status=1
        for attempt in 1 2 3; do
            set +e
            QBH_EXP0161_DUMP_CACHE_REMOTE="${remote_capture}" \
            QBH_EXP0161_DUMP_OUTPUT_REMOTE="${remote_capture}/block_output_u8.bin" \
                bash scripts/run_exp0161.sh \
                    "${length}" "${reconstruction}" 1 on \
                    > "${local_capture}/run.jsonl" \
                    2> "${local_capture}/run.stderr"
            capture_status=$?
            set -e
            if (( capture_status == 0 )); then
                break
            fi
            sleep 1
        done
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
