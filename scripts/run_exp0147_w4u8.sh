#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0147_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0147}"
model_root="${QBH_EXP0147_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0147}"
cell="${1:?cell required}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
dump_output="${QBH_DUMP_OUTPUT_PATH:-}"

case "${cell}" in
    prefill_m16)
        package="${model_root}/prefill_m16_w4u8_auditref"
        scan_mode=prefill; logical_m=16; past=0; capacity=16 ;;
    prefill_m32)
        package="${model_root}/prefill_m32_w4u8"
        scan_mode=prefill; logical_m=32; past=0; capacity=32 ;;
    prefill_m64)
        package="${model_root}/prefill_m64_w4u8"
        scan_mode=prefill; logical_m=64; past=0; capacity=64 ;;
    prefill_m128)
        package="${model_root}/prefill_m128_w4u8"
        scan_mode=prefill; logical_m=128; past=0; capacity=128 ;;
    decode_l64)
        package="${model_root}/decode_l64_w4u8_v2"
        scan_mode=decode; logical_m=1; past=64; capacity=65 ;;
    decode_l256)
        package="${model_root}/decode_l256_w4u8"
        scan_mode=decode; logical_m=1; past=256; capacity=257 ;;
    decode_l1024)
        package="${model_root}/decode_l1024_w4u8"
        scan_mode=decode; logical_m=1; past=1024; capacity=1025 ;;
    decode_l4096)
        package="${model_root}/decode_l4096_w4u8"
        scan_mode=decode; logical_m=1; past=4096; capacity=4097 ;;
    *)
        printf 'unknown EXP-0147 W4U8 cell: %s\n' "${cell}" >&2
        exit 2 ;;
esac

if [[ "${QBH_EXP0147_DEPLOY:-0}" == 1 ]]; then
    QBH_EXP0144_PACKAGE="${package}" EXP0144_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0144_block.sh" >/dev/null
fi

remote_dump=""
if [[ -n "${dump_output}" ]]; then
    remote_dump="QBH_DUMP_OUTPUT_PATH=${remote_root}/actual_block_output_u8.bin"
fi

"${adb_exe}" get-state >/dev/null
set +e
"${adb_exe}" shell \
    "cd ${remote_root} && ${remote_dump} QBH_SCAN_MODE=${scan_mode} QBH_LOGICAL_M=${logical_m} QBH_KV_CACHE_LENGTH=${past} QBH_KV_CACHE_CAPACITY=${capacity} QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} 2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"
run_status=$?
set -e

if [[ -n "${dump_output}" ]]; then
    mkdir -p "$(dirname "${dump_output}")"
    "${adb_exe}" pull \
        "${remote_root}/actual_block_output_u8.bin" \
        "$(wslpath -w "${dump_output}")" >/dev/null
else
    exit "${run_status}"
fi
