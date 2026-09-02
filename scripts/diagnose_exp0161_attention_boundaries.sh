#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0161/diagnostic_attention_boundaries}"
remote_root="/data/local/tmp/qwen3-block-htp/exp0161-l64-segmented"
remote_capture="${remote_root}/boundary_audit"

mkdir -p "${result_dir}"
"${adb_exe}" shell "mkdir -p ${remote_capture}" >/dev/null

set +e
"${adb_exe}" shell \
    "cd ${remote_root} && QBH_DUMP_ATTENTION_DIR=${remote_capture} QBH_DUMP_CACHE_DIR=${remote_capture} QBH_DUMP_OUTPUT_PATH=${remote_capture}/block_output_u8.bin QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8_segmented_v4 QBH_W4U8_DELTA_RECONSTRUCTION=serial QBH_SCAN_MODE=decode QBH_LOGICAL_M=1 QBH_KV_CACHE_LENGTH=64 QBH_KV_CACHE_CAPACITY=65 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 2 32 rms_rope_softmax on on fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2" \
    > "${result_dir}/run.jsonl" \
    2> "${result_dir}/run.stderr"
runner_status=$?
set -e
printf '%s\n' "${runner_status}" > "${result_dir}/runner_status"

for file in \
    actual_q_tiles_u8.bin \
    actual_k_tiles_u8.bin \
    actual_v_tiles_u8.bin \
    actual_score_tiles_u8.bin \
    actual_probability_tiles_u8.bin \
    actual_av_tiles_u8.bin \
    block_output_u8.bin \
    actual_kv_cache_k_u8.bin \
    actual_kv_cache_v_u8.bin; do
    "${adb_exe}" pull "${remote_capture}/${file}" \
        "$(wslpath -w "${result_dir}/${file}")" >/dev/null
done
