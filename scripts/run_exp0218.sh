#!/usr/bin/env bash
set -euo pipefail
recipe="${1:?recipe required}"
suite="${2:-}"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0218_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0218-${recipe}}"
evaluation_env=""
if [[ -n "$suite" ]]; then
    [[ "$suite" =~ ^[a-z0-9_-]+$ ]] || exit 2
    evaluation_env="QBH_EVAL_QUIET=1 QBH_EVAL_FILE=${remote_root}/${suite}.bin"
fi
case "$recipe" in
f16f16)
    mode=10; variant=F16F16; capacity=80
    recipe_env="QBH_KV_CACHE_LAYOUT=hmx_native_f16"
    args="2 32 hvx on off fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
    ;;
w4f16)
    mode=7; variant=W4F16; capacity=80
    recipe_env="QBH_KV_CACHE_LAYOUT=hmx_native_f16 QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
    args="4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
    ;;
w4u8)
    mode=9; variant=W4U8; capacity=257
    recipe_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8_segmented_vtcm_k7_session_v9 QBH_W4U8_PREFILL_CACHE_MODE=reuse QBH_W4U8_DELTA_RECONSTRUCTION=serial QBH_W4U8_DECODE_SOFTMAX=hvx_tile4 QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES=32 QBH_W4U8_DECODE_O_BATCH_N_TILES=16 QBH_W4U8_DECODE_AV_REQUANT_ROWS=4 QBH_W4U8_DECODE_AV_PADDING_POISON=0 QBH_W4U8_DECODE_COMMON_OP_ROWS=4 QBH_W4U8_DECODE_COMMON_PADDING_POISON=0 QBH_W4U8_DECODE_QK_NORM_ROPE_ROWS=4 QBH_W4U8_DECODE_QK_PADDING_POISON=0 QBH_W4U8_DECODE_PROJECTION_MODE=direct_n QBH_W4U8_DECODE_DIRECT_N_MASK=63 QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES=32 QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS=1 QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH=1 QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM=1 QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES=16 QBH_W4U8_DECODE_DIRECT_N_Q_BATCH_N_TILES=32 QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES=8 QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA=1 QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA=1 QBH_W4U8_DECODE_SWIGLU_ROWS=4 QBH_W4U8_DECODE_SWIGLU_PADDING_POISON=0"
    args="2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"
    ;;
*) exit 2 ;;
esac
"$adb_exe" shell "cd ${remote_root} && ${evaluation_env} ${recipe_env} QBH_GENERATION_SEQUENCE=${mode} QBH_GENERATION_STEPS=16 QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=${capacity} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} 1 ${args}"
