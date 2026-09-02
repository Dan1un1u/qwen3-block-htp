#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0155_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0155/layer14_hmx_cache_v1}"
layout="${1:-hmx_native_u8}"
mode="${2:-replay}"
remote_root="${EXP0155_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0155-${layout}}"
capture_root="${QBH_REPLAY_CAPTURE_ROOT:-}"

case "${layout}" in
    row_major|hmx_native_u8) ;;
    *) printf 'unknown cache layout: %s\n' "${layout}" >&2; exit 2 ;;
esac
case "${mode}" in
    layout) stage_env="QBH_LAYOUT_ONLY=1" ;;
    replay) stage_env="" ;;
    *) printf 'unknown mode: %s\n' "${mode}" >&2; exit 2 ;;
esac

runtime_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=${layout}"
runtime_args="2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"

test -f "${package}/manifest_exp0155.json"
if [[ "${QBH_EXP0155_DEPLOY:-0}" == 1 ]]; then
    EXP0147_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
fi

"${adb_exe}" get-state >/dev/null
capture_env=""
if [[ "${mode}" == replay && -n "${capture_root}" ]]; then
    mkdir -p "${capture_root}"
    "${adb_exe}" shell \
        "mkdir -p ${remote_root}/replay_capture && rm -f ${remote_root}/replay_capture/*"
    capture_env="QBH_REPLAY_DUMP_DIR=${remote_root}/replay_capture"
fi
set +e
"${adb_exe}" shell \
    "cd ${remote_root} && ${runtime_env} ${capture_env} ${stage_env} QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=72 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 ${runtime_args}"
run_status=$?
set -e
if [[ "${mode}" == replay && -n "${capture_root}" ]]; then
    "${adb_exe}" pull "${remote_root}/replay_capture/." \
        "$(wslpath -w "${capture_root}")" >/dev/null
fi
exit "${run_status}"
