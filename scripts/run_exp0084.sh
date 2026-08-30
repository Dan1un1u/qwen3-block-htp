#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_remote="${EXP0084_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0084-fp16}"
u8_remote="${EXP0084_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0084-u8}"
variant="${1:-W4F16}"
schedule="${2:-control}"
repeat_count="${3:-1}"
attribution_mode="${4:-on}"
audit_mode="${5:-off}"
fp16_norm_rows="${EXP0084_FP16_NORM_ROWS:-4}"
fp16_norm_contexts="${EXP0084_FP16_NORM_CONTEXTS:-4}"

case "${schedule}" in
    control)
        fp16_schedule=control
        ;;
    stage_a|qk_head_pairs)
        fp16_schedule=qk_head_pairs
        ;;
    stage_b|input_norm_pool)
        fp16_schedule=input_norm_pool
        ;;
    stage_c|post_norm_pool)
        fp16_schedule=post_norm_pool
        ;;
    candidate|selected|input_norm_pool_post_norm_pool)
        fp16_schedule=input_norm_pool_post_norm_pool
        ;;
    *)
        printf 'invalid EXP-0084 schedule: %s\n' "${schedule}" >&2
        exit 2
        ;;
esac

case "${variant}" in
    F16F16)
        remote_root="${fp16_remote}"
        runtime_args="2 32 hvx ${attribution_mode} ${audit_mode} fused gate8 control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar ${fp16_schedule} ${fp16_norm_rows} ${fp16_norm_contexts}"
        ;;
    W4F16)
        remote_root="${fp16_remote}"
        runtime_args="3 32 hvx ${attribution_mode} ${audit_mode} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar ${fp16_schedule} ${fp16_norm_rows} ${fp16_norm_contexts}"
        ;;
    W4U8)
        if [[ "${schedule}" != "control" ]]; then
            printf 'W4U8 is immutable in EXP-0084; use control\n' >&2
            exit 2
        fi
        remote_root="${u8_remote}"
        runtime_args="2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused_pool6 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4"
        ;;
    *)
        printf 'usage: %s F16F16|W4F16|W4U8 [control|stage_a|stage_b|stage_c] [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
