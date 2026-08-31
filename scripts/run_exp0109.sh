#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_remote="${EXP0109_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0109-fp16}"
u8_remote="${EXP0109_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0109-u8}"
cell="${1:-frozen_f16f16}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
u8_attention="u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream"

case "${cell}" in
    control_f16f16|frozen_f16f16)
        remote_root="${fp16_remote}"
        variant=F16F16
        projection_mode=gate8
        [[ "${cell}" == frozen_f16f16 ]] && \
            projection_mode=gate8_interleaved
        runtime_args="2 32 hvx ${attribution_mode} ${audit_mode} fused ${projection_mode} control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    fair_w4f16)
        remote_root="${fp16_remote}"
        variant=W4F16
        runtime_args="3 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    fair_w4u8|fastest_w4u8)
        remote_root="${u8_remote}"
        variant=W4U8
        residual_mode=fused_pool6
        attention_pipeline="${u8_attention}"
        down_batch_outputs=1
        qk_pair_kernel=0
        if [[ "${cell}" == fastest_w4u8 ]]; then
            residual_mode=fused_pool6_shuffle4
            attention_pipeline="${u8_attention}_softmax_shuffle4"
            down_batch_outputs=4
            qk_pair_kernel=1
        fi
        runtime_args="2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} ${residual_mode} serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 ${attention_pipeline} 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 ${down_batch_outputs} ${qk_pair_kernel}"
        ;;
    *)
        printf 'usage: %s control_f16f16|frozen_f16f16|fair_w4f16|fair_w4u8|fastest_w4u8 [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
