#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${EXP0167_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0167-w4u8}"
audit="${QBH_GENERATION_BOUNDARY_AUDIT:-0}"
audit_dir="${QBH_GENERATION_AUDIT_DIR:-}"
remote_audit_dir="${remote_root}/generation-audit"
runtime_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8_segmented_v4 QBH_W4U8_PREFILL_CACHE_MODE=reuse QBH_W4U8_DELTA_RECONSTRUCTION=serial"
runtime_args="2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"
stage_env=""
if [[ "${QBH_LAYOUT_ONLY:-0}" == 1 ]]; then
    stage_env="QBH_LAYOUT_ONLY=1"
elif [[ "${QBH_MAP_ONLY:-0}" == 1 ]]; then
    stage_env="QBH_MAP_ONLY=1"
fi

if [[ "${QBH_EXP0167_DEPLOY:-0}" == 1 ]]; then
    "${project_root}/scripts/deploy_exp0167.sh"
fi
"${adb_exe}" get-state >/dev/null
if [[ "${audit}" == 1 ]]; then
    test -n "${audit_dir}"
    mkdir -p "${audit_dir}"
    "${adb_exe}" shell "mkdir -p ${remote_audit_dir} && rm -f ${remote_audit_dir}/generation_hidden_step*_u8.bin"
fi
"${adb_exe}" shell \
    "cd ${remote_root} && ${runtime_env} ${stage_env} QBH_GENERATION_BOUNDARY_AUDIT=${audit} QBH_GENERATION_AUDIT_DIR=${remote_audit_dir} QBH_GENERATION_SEQUENCE=8 QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=80 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 ${runtime_args}" \
    | if [[ -n "${audit_dir}" ]]; then tee "${audit_dir}/device.jsonl"; else cat; fi
if [[ "${audit}" == 1 ]]; then
    "${adb_exe}" pull "${remote_audit_dir}/." "$(wslpath -w "${audit_dir}")" >/dev/null
fi
