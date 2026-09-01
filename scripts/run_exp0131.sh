#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0131_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0131-u8}"
cell="${1:-control}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
tail_prep=control
u8_attention="u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4"

case "${cell}" in
    control)
        ;;
    main_drain)
        tail_prep=main_drain
        ;;
    *)
        printf 'usage: %s control|main_drain [repeat] [attribution] [audit]\n' "$0" >&2
        exit 2
        ;;
esac

runtime_args="2 32 rms_rope_softmax ${attribution_mode} ${audit_mode} fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 ${u8_attention} 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 1"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_W4U8_QKV_TAIL_PREP=${tail_prep} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} ${runtime_args}"
