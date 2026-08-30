#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_remote="${EXP0085_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0085-fp16}"
u8_remote="${EXP0085_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0085-u8}"
variant="${1:-W4U8}"
repeat_count="${2:-1}"
qkv_schedule="${3:-control}"
attribution="${4:-on}"
audit="${5:-off}"

if [[ "${qkv_schedule}" != "control" &&
      "${qkv_schedule}" != "group_major" &&
      "${qkv_schedule}" != "group_major2" &&
      "${qkv_schedule}" != "group_major4" &&
      "${qkv_schedule}" != "q_prefix4_k_all" &&
      "${qkv_schedule}" != "q_prefix6_k_all" ]]; then
    printf 'invalid QKV schedule: %s\n' "${qkv_schedule}" >&2
    exit 2
fi
if [[ "${attribution}" != "on" && "${attribution}" != "off" ]]; then
    printf 'invalid attribution mode: %s\n' "${attribution}" >&2
    exit 2
fi
if [[ "${audit}" != "on" && "${audit}" != "off" ]]; then
    printf 'invalid numerical audit mode: %s\n' "${audit}" >&2
    exit 2
fi

case "${variant}" in
    F16F16)
        remote_root="${fp16_remote}"
        fp16_boundary="norms"
        if [[ "${qkv_schedule}" != "control" ]]; then
            fp16_boundary="qkv_norms"
        fi
        runtime_args="2 32 hvx ${attribution} ${audit} fused gate8 control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 ${fp16_boundary} serial scalar input_norm_pool_post_norm_pool 4 3"
        ;;
    W4F16)
        remote_root="${fp16_remote}"
        fp16_boundary="norms"
        if [[ "${qkv_schedule}" != "control" ]]; then
            fp16_boundary="qkv_norms"
        fi
        runtime_args="3 32 hvx ${attribution} ${audit} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 ${fp16_boundary} serial scalar input_norm_pool_post_norm_pool 4 3"
        ;;
    W4U8)
        remote_root="${u8_remote}"
        runtime_args="2 32 rms_rope_softmax ${attribution} ${audit} fused_pool6 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4"
        ;;
    *)
        printf 'invalid variant: %s\n' "${variant}" >&2
        exit 2
        ;;
esac

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root} && QBH_QKV_SCHEDULE=${qkv_schedule} QBH_QKV_TIMELINE=0 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
