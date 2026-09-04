#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0173_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0173-w4u8}"
generation_steps="${QBH_EXP0173_GENERATION_STEPS:-193}"
group_tiles="${QBH_EXP0173_LM_HEAD_GROUP_TILES:-8}"
o_batch_tiles="${QBH_EXP0176_O_BATCH_TILES:-4}"
av_requant_rows="${QBH_EXP0177_AV_REQUANT_ROWS:-64}"
av_padding_poison="${QBH_EXP0177_AV_PADDING_POISON:-0}"
common_op_rows="${QBH_EXP0178_COMMON_OP_ROWS:-64}"
common_padding_poison="${QBH_EXP0178_COMMON_PADDING_POISON:-0}"
qk_norm_rope_rows="${QBH_EXP0179_QK_NORM_ROPE_ROWS:-64}"
qk_padding_poison="${QBH_EXP0179_QK_PADDING_POISON:-0}"
decode_projection_mode="${QBH_EXP0188_DECODE_PROJECTION_MODE:-expanded_s8}"
decode_direct_n_mask="${QBH_W4U8_DECODE_DIRECT_N_MASK:-0}"
decode_direct_n_gate_up_batch_tiles="${QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES:-4}"
decode_direct_n_down_batch_tiles="${QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES:-2}"
decode_swiglu_rows="${QBH_EXP0189_SWIGLU_ROWS:-64}"
decode_swiglu_padding_poison="${QBH_EXP0189_SWIGLU_PADDING_POISON:-0}"
kv_cache_layout="${QBH_EXP0183_KV_CACHE_LAYOUT:-${QBH_EXP0182_KV_CACHE_LAYOUT:-${QBH_EXP0181_KV_CACHE_LAYOUT:-${QBH_EXP0180_KV_CACHE_LAYOUT:-hmx_native_u8_segmented_v4}}}}"
audit="${QBH_GENERATION_BOUNDARY_AUDIT:-0}"
audit_dir="${QBH_GENERATION_AUDIT_DIR:-}"
layout_only="${QBH_LAYOUT_ONLY:-0}"
remote_audit_dir="${remote_root}/generation-audit-batch${group_tiles}"
runtime_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=${kv_cache_layout} QBH_W4U8_PREFILL_CACHE_MODE=reuse QBH_W4U8_DELTA_RECONSTRUCTION=serial QBH_W4U8_DECODE_SOFTMAX=hvx_tile4 QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES=${group_tiles} QBH_W4U8_DECODE_O_BATCH_N_TILES=${o_batch_tiles} QBH_W4U8_DECODE_AV_REQUANT_ROWS=${av_requant_rows} QBH_W4U8_DECODE_AV_PADDING_POISON=${av_padding_poison} QBH_W4U8_DECODE_COMMON_OP_ROWS=${common_op_rows} QBH_W4U8_DECODE_COMMON_PADDING_POISON=${common_padding_poison} QBH_W4U8_DECODE_QK_NORM_ROPE_ROWS=${qk_norm_rope_rows} QBH_W4U8_DECODE_QK_PADDING_POISON=${qk_padding_poison} QBH_W4U8_DECODE_PROJECTION_MODE=${decode_projection_mode} QBH_W4U8_DECODE_DIRECT_N_MASK=${decode_direct_n_mask} QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES=${decode_direct_n_gate_up_batch_tiles} QBH_W4U8_DECODE_SWIGLU_ROWS=${decode_swiglu_rows} QBH_W4U8_DECODE_SWIGLU_PADDING_POISON=${decode_swiglu_padding_poison} QBH_LAYOUT_ONLY=${layout_only}"
down_batch_outputs=4
if [[ "${decode_projection_mode}" == direct_n ]]; then
    down_batch_outputs="${decode_direct_n_down_batch_tiles}"
fi
runtime_args="2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 ${down_batch_outputs} 2"

case "${group_tiles}" in
8|16) ;;
*) printf 'invalid EXP-0173 LM-head group tiles: %s\n' "${group_tiles}" >&2; exit 2 ;;
esac
case "${o_batch_tiles}" in
4|8) ;;
*) printf 'invalid EXP-0176 O batch tiles: %s\n' "${o_batch_tiles}" >&2; exit 2 ;;
esac
case "${av_requant_rows}" in
4|64) ;;
*) printf 'invalid EXP-0177 AV requant rows: %s\n' "${av_requant_rows}" >&2; exit 2 ;;
esac
case "${av_padding_poison}" in
0|1) ;;
*) printf 'invalid EXP-0177 AV padding poison: %s\n' "${av_padding_poison}" >&2; exit 2 ;;
esac
if [[ "${av_padding_poison}" == 1 && "${av_requant_rows}" != 4 ]]; then
    printf 'AV padding poison requires four-row requantization\n' >&2
    exit 2
fi
case "${common_op_rows}" in
4|64) ;;
*) printf 'invalid EXP-0178 common-op rows: %s\n' "${common_op_rows}" >&2; exit 2 ;;
esac
case "${common_padding_poison}" in
0|1) ;;
*) printf 'invalid EXP-0178 common padding poison: %s\n' "${common_padding_poison}" >&2; exit 2 ;;
esac
case "${qk_norm_rope_rows}" in
4|64) ;;
*) printf 'invalid EXP-0179 Q/K Norm-RoPE rows: %s\n' "${qk_norm_rope_rows}" >&2; exit 2 ;;
esac
case "${qk_padding_poison}" in
0|1) ;;
*) printf 'invalid EXP-0179 Q/K padding poison: %s\n' "${qk_padding_poison}" >&2; exit 2 ;;
esac
if [[ "${qk_padding_poison}" == 1 && "${qk_norm_rope_rows}" != 4 ]]; then
    printf 'Q/K padding poison requires four-row preparation\n' >&2
    exit 2
fi
case "${kv_cache_layout}" in
hmx_native_u8_segmented_v4|hmx_native_u8_segmented_quartet_v5|hmx_native_u8_segmented_attention_publish_v6|hmx_native_u8_segmented_vtcm_tail_v7|hmx_native_u8_segmented_vtcm_k7_v8|hmx_native_u8_segmented_vtcm_k7_session_v9) ;;
*) printf 'invalid EXP-0180/0181/0182/0183 KV-cache layout: %s\n' "${kv_cache_layout}" >&2; exit 2 ;;
esac
case "${decode_projection_mode}" in
expanded_s8|direct_n) ;;
*) printf 'invalid EXP-0188 decode projection mode: %s\n' "${decode_projection_mode}" >&2; exit 2 ;;
esac
case "${decode_swiglu_rows}" in
4|64) ;;
*) printf 'invalid EXP-0189 SwiGLU rows: %s\n' "${decode_swiglu_rows}" >&2; exit 2 ;;
esac
case "${decode_direct_n_gate_up_batch_tiles}" in
4|8|16) ;;
*) printf 'invalid direct-n Gate/Up batch tiles: %s\n' "${decode_direct_n_gate_up_batch_tiles}" >&2; exit 2 ;;
esac
case "${decode_direct_n_down_batch_tiles}" in
2|4) ;;
*) printf 'invalid direct-n Down batch tiles: %s\n' "${decode_direct_n_down_batch_tiles}" >&2; exit 2 ;;
esac
case "${decode_swiglu_padding_poison}" in
0|1) ;;
*) printf 'invalid EXP-0189 SwiGLU padding poison: %s\n' "${decode_swiglu_padding_poison}" >&2; exit 2 ;;
esac
if [[ "${decode_swiglu_rows}" == 4 &&
      ("${decode_projection_mode}" != direct_n ||
       $((decode_direct_n_mask & 4)) -eq 0) ]]; then
    printf 'four-row SwiGLU requires direct-n MLP\n' >&2
    exit 2
fi
if [[ "${decode_swiglu_padding_poison}" == 1 &&
      "${decode_swiglu_rows}" != 4 ]]; then
    printf 'SwiGLU padding poison requires four-row SwiGLU\n' >&2
    exit 2
fi
if [[ "${common_padding_poison}" == 1 && "${common_op_rows}" != 4 ]]; then
    printf 'common padding poison requires four-row common ops\n' >&2
    exit 2
fi
if [[ "${QBH_EXP0173_DEPLOY:-0}" == 1 ]]; then
    "${project_root}/scripts/deploy_exp0173.sh"
fi
"${adb_exe}" get-state >/dev/null
if [[ "${audit}" == 1 ]]; then
    test -n "${audit_dir}"
    mkdir -p "${audit_dir}"
    "${adb_exe}" shell "mkdir -p ${remote_audit_dir} && rm -f ${remote_audit_dir}/generation_hidden_step*_u8.bin"
fi
"${adb_exe}" shell \
    "cd ${remote_root} && ${runtime_env} QBH_GENERATION_BOUNDARY_AUDIT=${audit} QBH_GENERATION_AUDIT_DIR=${remote_audit_dir} QBH_GENERATION_SEQUENCE=9 QBH_GENERATION_STEPS=${generation_steps} QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=257 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 ${runtime_args}" \
    | if [[ -n "${audit_dir}" ]]; then tee "${audit_dir}/device.jsonl"; else cat; fi
if [[ "${audit}" == 1 ]]; then
    "${adb_exe}" pull "${remote_audit_dir}/." \
        "$(wslpath -w "${audit_dir}")" >/dev/null
fi
