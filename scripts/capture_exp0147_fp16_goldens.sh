#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${QBH_EXP0147_PYTHON:-/home/daniuniu/mllm-quant-venv/bin/python}"
recipe="${1:?f16f16 or w4f16 required}"
model_root="${QBH_EXP0147_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0147}"
result_root="${QBH_EXP0147_RESULT_ROOT:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/${recipe}}"

case "${recipe}" in
    f16f16|w4f16) ;;
    *) printf 'unknown recipe: %s\n' "${recipe}" >&2; exit 2 ;;
esac

cells=(
    prefill_m16 prefill_m32 prefill_m64 prefill_m128
    decode_l64 decode_l256 decode_l1024 decode_l4096
)

for cell in "${cells[@]}"; do
    package="${model_root}/${cell}_${recipe}"
    capture="${result_root}/${cell}"
    mkdir -p "${capture}"

    # The first execution is expected to fail the exact device-golden check
    # for a newly generated FP16 shape package.  Captures must still exist and
    # pass the independent numerical audits before they may be promoted.
    set +e
    QBH_EXP0147_DEPLOY=1 QBH_EXP0147_CAPTURE_ROOT="${capture}" \
        "${project_root}/scripts/run_exp0147_fp16.sh" \
            "${recipe}" "${cell}" 1 on on \
            >"${capture}/capture_run.json"
    capture_status=$?
    set -e
    printf '%s\n' "${capture_status}" >"${capture}/capture_run.status"

    for required in actual_block_output_f16.bin \
                    actual_kv_cache_k_f16.bin actual_kv_cache_v_f16.bin \
                    actual_scan_q_f16.bin actual_scan_attention_f16.bin \
                    actual_scan_o_projection_f16.bin; do
        test -s "${capture}/${required}"
    done

    set +e
    "${python_bin}" "${project_root}/scripts/verify_exp0147_fp16_capture.py" \
        --package "${package}" --capture "${capture}" --recipe "${recipe}" \
        --output "${capture}/independent_capture_audit.json" \
        >"${capture}/independent_capture_audit.stdout"
    independent_status=$?
    set -e
    printf '%s\n' "${independent_status}" \
        >"${capture}/independent_capture_audit.status"

    "${python_bin}" "${project_root}/scripts/verify_exp0147_fp16_attention.py" \
        --package "${package}" --capture "${capture}" \
        --output "${capture}/attention_boundary_audit.json" \
        >"${capture}/attention_boundary_audit.stdout"

    "${python_bin}" "${project_root}/scripts/promote_exp0147_fp16_golden.py" \
        --package "${package}" --capture "${capture}" \
        --recipe "${recipe}" --promote-output \
        >"${capture}/promotion.json"
done
