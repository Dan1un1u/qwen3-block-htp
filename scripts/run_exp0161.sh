#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
past_length="${1:?past length required}"
reconstruction="${2:?serial, direct, pipeline, or segmented required}"
repeat_count="${3:-1}"
attribution_mode="${4:-on}"
capacity=$((past_length + 1))
remote_suffix="delta"
cache_layout="hmx_native_u8_delta"
delta_reconstruction="${reconstruction}"
dump_env=""

case "${past_length}" in 64|256|1024|4096) ;; *) exit 2 ;; esac
case "${reconstruction}" in
serial|direct|pipeline) ;;
segmented)
    remote_suffix="segmented"
    cache_layout="hmx_native_u8_segmented_v4"
    delta_reconstruction="serial"
    ;;
*) exit 2 ;;
esac
remote_root="${EXP0161_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0161-l${past_length}-${remote_suffix}}"

if [[ -n "${QBH_EXP0161_DUMP_CACHE_REMOTE:-}" ]]; then
    "${adb_exe}" shell "mkdir -p ${QBH_EXP0161_DUMP_CACHE_REMOTE}"
    dump_env="QBH_DUMP_CACHE_DIR=${QBH_EXP0161_DUMP_CACHE_REMOTE}"
fi
if [[ -n "${QBH_EXP0161_DUMP_OUTPUT_REMOTE:-}" ]]; then
    dump_env="${dump_env} QBH_DUMP_OUTPUT_PATH=${QBH_EXP0161_DUMP_OUTPUT_REMOTE}"
fi

runtime_args="2 32 rms_rope_softmax ${attribution_mode} off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root} && ${dump_env} QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=${cache_layout} QBH_W4U8_DELTA_RECONSTRUCTION=${delta_reconstruction} QBH_SCAN_MODE=decode QBH_LOGICAL_M=1 QBH_KV_CACHE_LENGTH=${past_length} QBH_KV_CACHE_CAPACITY=${capacity} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 ${repeat_count} ${runtime_args}"
