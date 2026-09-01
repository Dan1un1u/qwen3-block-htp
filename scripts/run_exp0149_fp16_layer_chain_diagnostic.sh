#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python3}"
recipe="${1:?f16f16 or w4f16 required}"
stamp="${QBH_DIAGNOSTIC_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
model_run="${QBH_DIAGNOSTIC_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149-repair/${stamp}_${recipe}}"
result_run="${QBH_DIAGNOSTIC_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0149-repair/${stamp}_${recipe}}"
remote_base="${QBH_DIAGNOSTIC_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0149-repair-${recipe}}"
source_root="${QBH_EXP0149_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149}"

case "${recipe}" in
    f16f16)
        variant=F16F16
        runtime_env=""
        runtime_args="2 32 hvx on off fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    w4f16)
        variant=W4F16
        runtime_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
        runtime_args="4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    *)
        printf 'unknown recipe: %s\n' "${recipe}" >&2
        exit 2
        ;;
esac

mkdir -p "${model_run}" "${result_run}"
input_path="${source_root}/${recipe}/block_input_f16.bin"
overall_status=0

for layer in 13 14 15; do
    package="${model_run}/layer${layer}"
    capture="${result_run}/layer${layer}"
    remote_root="${remote_base}-layer${layer}"
    mkdir -p "${capture}"
    "${python_exe}" \
        "${project_root}/scripts/prepare_exp0149_fp16_layer_diagnostic.py" \
        --recipe "${recipe}" --layer "${layer}" \
        --input "${input_path}" --output "${package}" \
        >"${capture}/prepare.json"

    EXP0147_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
    "${adb_exe}" shell "mkdir -p ${remote_root}/capture"

    set +e
    "${adb_exe}" shell \
        "cd ${remote_root} && ${runtime_env} QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=64 QBH_DUMP_OUTPUT_PATH=${remote_root}/capture/actual_block_output_f16.bin QBH_DUMP_CACHE_DIR=${remote_root}/capture LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} 1 ${runtime_args}" \
        >"${capture}/device.jsonl" 2>"${capture}/device.stderr"
    device_status=$?
    set -e
    "${adb_exe}" pull "${remote_root}/capture/." \
        "$(wslpath -w "${capture}")" >/dev/null

    set +e
    "${python_exe}" \
        "${project_root}/scripts/verify_exp0147_fp16_capture.py" \
        --package "${package}" --capture "${capture}" \
        --recipe "${recipe}" --output "${capture}/verification.json" \
        >"${capture}/verification.stdout"
    verify_status=$?
    set -e
    if [[ ${device_status} -ne 0 || ${verify_status} -ne 0 ]]; then
        overall_status=1
    fi
    input_path="${capture}/actual_block_output_f16.bin"
done

printf '{"experiment":"EXP-0149","role":"fp16_layer_chain_diagnostic","recipe":"%s","model_root":"%s","result_root":"%s","all_local_gates_pass":%s}\n' \
    "${recipe}" "${model_run}" "${result_run}" \
    "$([[ ${overall_status} -eq 0 ]] && printf true || printf false)"
exit "${overall_status}"
