#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
model_root="${QBH_EXP0147_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0147}"
remote_root="${EXP0147_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0147-fp16}"
recipe="${1:?f16f16 or w4f16 required}"
cell="${2:?cell required}"
repeat_count="${3:-1}"
attribution_mode="${4:-on}"
audit_mode="${5:-off}"
dump_root="${QBH_EXP0147_CAPTURE_ROOT:-}"

case "${cell}" in
    prefill_m16) scan_mode=prefill; logical_m=16; past=0; capacity=16 ;;
    prefill_m32) scan_mode=prefill; logical_m=32; past=0; capacity=32 ;;
    prefill_m64) scan_mode=prefill; logical_m=64; past=0; capacity=64 ;;
    prefill_m128) scan_mode=prefill; logical_m=128; past=0; capacity=128 ;;
    decode_l64) scan_mode=decode; logical_m=1; past=64; capacity=65 ;;
    decode_l256) scan_mode=decode; logical_m=1; past=256; capacity=257 ;;
    decode_l1024) scan_mode=decode; logical_m=1; past=1024; capacity=1025 ;;
    decode_l4096) scan_mode=decode; logical_m=1; past=4096; capacity=4097 ;;
    *) printf 'unknown EXP-0147 cell: %s\n' "${cell}" >&2; exit 2 ;;
esac

case "${recipe}" in
    f16f16)
        variant=F16F16
        package="${model_root}/${cell}_f16f16"
        runtime_env=""
        runtime_args="2 32 hvx ${attribution_mode} ${audit_mode} fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    w4f16)
        variant=W4F16
        package="${model_root}/${cell}_w4f16"
        # Preserve the selected EXP-0140 physical kernel identity.  These are
        # recipe-specific scheduling controls, not scan variables.
        runtime_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
        runtime_args="4 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    *) printf 'unknown recipe: %s\n' "${recipe}" >&2; exit 2 ;;
esac

if [[ "${QBH_EXP0147_DEPLOY:-0}" == 1 ]]; then
    EXP0147_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
fi

remote_dump_env=""
if [[ -n "${dump_root}" ]]; then
    mkdir -p "${dump_root}"
    "${adb_exe}" shell "mkdir -p ${remote_root}/capture"
    "${adb_exe}" shell \
        "rm -f ${remote_root}/capture/actual_block_output_f16.bin ${remote_root}/capture/actual_kv_cache_k_f16.bin ${remote_root}/capture/actual_kv_cache_v_f16.bin ${remote_root}/capture/actual_scan_q_f16.bin ${remote_root}/capture/actual_scan_attention_f16.bin ${remote_root}/capture/actual_scan_o_projection_f16.bin"
    remote_dump_env="QBH_DUMP_OUTPUT_PATH=${remote_root}/capture/actual_block_output_f16.bin QBH_DUMP_CACHE_DIR=${remote_root}/capture QBH_DUMP_ATTENTION_DIR=${remote_root}/capture"
fi

"${adb_exe}" get-state >/dev/null
set +e
"${adb_exe}" shell \
    "cd ${remote_root} && ${remote_dump_env} ${runtime_env} QBH_SCAN_MODE=${scan_mode} QBH_LOGICAL_M=${logical_m} QBH_KV_CACHE_LENGTH=${past} QBH_KV_CACHE_CAPACITY=${capacity} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
run_status=$?
set -e

if [[ -n "${dump_root}" ]]; then
    for file in actual_block_output_f16.bin \
                actual_kv_cache_k_f16.bin actual_kv_cache_v_f16.bin \
                actual_scan_q_f16.bin actual_scan_attention_f16.bin \
                actual_scan_o_projection_f16.bin; do
        "${adb_exe}" pull "${remote_root}/capture/${file}" \
            "$(wslpath -w "${dump_root}/${file}")" >/dev/null
    done
fi
exit "${run_status}"
