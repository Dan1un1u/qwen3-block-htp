#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_remote="${EXP0106_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0106-fp16}"
u8_remote="${EXP0106_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0106-u8}"
cell="${1:-fair_w4u8}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
control_attention="u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream"

case "${cell}" in
    fair_f16f16)
        remote_root="${fp16_remote}"
        variant=F16F16
        runtime_args="2 32 hvx ${attribution_mode} ${audit_mode} fused gate8 control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 4 1 0"
        ;;
    fair_w4f16)
        remote_root="${fp16_remote}"
        variant=W4F16
        runtime_args="3 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 4 1 0"
        ;;
    fair_w4u8|down4|down4_softmax|down4_softmax_residual|integrated)
        remote_root="${u8_remote}"
        variant=W4U8
        residual_mode=fused_pool6
        attention_pipeline="${control_attention}"
        down_batch_outputs=1
        qk_pair_kernel=0
        if [[ "${cell}" != fair_w4u8 ]]; then
            down_batch_outputs=4
        fi
        if [[ "${cell}" == down4_softmax || \
              "${cell}" == down4_softmax_residual || \
              "${cell}" == integrated ]]; then
            attention_pipeline="${control_attention}_softmax_shuffle4"
        fi
        if [[ "${cell}" == down4_softmax_residual || \
              "${cell}" == integrated ]]; then
            residual_mode=fused_pool6_shuffle4
        fi
        if [[ "${cell}" == integrated ]]; then
            qk_pair_kernel=1
        fi
        runtime_args="2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} ${residual_mode} serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 ${attention_pipeline} 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 ${down_batch_outputs} ${qk_pair_kernel}"
        ;;
    *)
        printf 'usage: %s fair_f16f16|fair_w4f16|fair_w4u8|down4|down4_softmax|down4_softmax_residual|integrated [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
