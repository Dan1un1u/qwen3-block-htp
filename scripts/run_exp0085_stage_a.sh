#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
fp16_remote="${EXP0085_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0085-fp16}"
u8_remote="${EXP0085_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0085-u8}"
variant="${1:-W4U8}"
repeat_count="${2:-1}"
timeline_output="${3:-}"
remote_timeline="/data/local/tmp/qwen3-block-htp/exp0085-${variant,,}-qkv-timeline.json"

if [[ -z "${timeline_output}" ]]; then
    printf 'usage: %s F16F16|W4F16|W4U8 repeat timeline-output\n' "$0" >&2
    exit 2
fi

case "${variant}" in
    F16F16)
        remote_root="${fp16_remote}"
        runtime_args="2 32 hvx on on fused gate8 control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3"
        ;;
    W4F16)
        remote_root="${fp16_remote}"
        runtime_args="3 32 hvx on on fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3"
        ;;
    W4U8)
        remote_root="${u8_remote}"
        runtime_args="2 32 rms_rope_softmax on on fused_pool6 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4"
        ;;
    *)
        printf 'invalid variant: %s\n' "${variant}" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${timeline_output}")"
"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell rm -f "${remote_timeline}"
"${adb_exe}" shell \
    "cd ${remote_root} && QBH_DUMP_QKV_TIMELINE_PATH=${remote_timeline} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${runtime_args}"
"${adb_exe}" exec-out cat "${remote_timeline}" >"${timeline_output}"
