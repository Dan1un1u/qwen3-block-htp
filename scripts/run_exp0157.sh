#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0157_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0156/w4u8_hmx_native_v1}"
prefill_mode="${1:-reuse}"
stage="${2:-replay}"
remote_root="${EXP0157_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0157-prefill-carrier-reuse}"

case "${prefill_mode}" in
    duplicate|reuse) ;;
    *) printf 'unknown prefill cache mode: %s\n' "${prefill_mode}" >&2; exit 2 ;;
esac
case "${stage}" in
    layout) stage_env="QBH_LAYOUT_ONLY=1" ;;
    map) stage_env="QBH_MAP_ONLY=1" ;;
    replay) stage_env="" ;;
    *) printf 'unknown stage: %s\n' "${stage}" >&2; exit 2 ;;
esac

runtime_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8 QBH_W4U8_PREFILL_CACHE_MODE=${prefill_mode}"
runtime_args="2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"

test -f "${package}/manifest.json"
if [[ "${QBH_EXP0157_DEPLOY:-0}" == 1 ]]; then
    EXP0157_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0157.sh" "${package}"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root} && ${runtime_env} ${stage_env} QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=72 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 ${runtime_args}"
