#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
model_root="${QBH_EXP0149_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149}"
recipe="${1:?f16f16, w4f16, or w4u8 required}"
remote_root="${EXP0149_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0149-${recipe}}"
package="${QBH_EXP0149_PACKAGE:-${model_root}/${recipe}}"
capture_root="${QBH_REPLAY_CAPTURE_ROOT:-}"
numerical_audit="${QBH_EXP0149_NUMERICAL_AUDIT:-off}"

case "${recipe}" in
    f16f16)
        variant=F16F16
        runtime_env=""
        runtime_args="2 32 hvx on ${numerical_audit} fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    w4f16)
        variant=W4F16
        runtime_env="QBH_QKV_SCHEDULE=control QBH_W4F16_GROUP_FENCE=join_only_down QBH_W4F16_EXPAND_CLAIM_REGIONS=1 QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER=1 QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER=1 QBH_W4F16_GATE_UP_STREAM_GROUP_TILES=4"
        runtime_args="4 32 hvx on ${numerical_audit} fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0"
        ;;
    w4u8)
        variant=W4U8
        runtime_env="QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3"
        runtime_args="2 32 rms_rope_softmax on ${numerical_audit} fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2"
        ;;
    *)
        printf 'unknown recipe: %s\n' "${recipe}" >&2
        exit 2
        ;;
esac

test -f "${package}/manifest.json"
if [[ "${QBH_EXP0149_DEPLOY:-0}" == 1 ]]; then
    EXP0147_REMOTE_ROOT="${remote_root}" \
        "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
fi

"${adb_exe}" get-state >/dev/null
capture_env=""
if [[ -n "${capture_root}" ]]; then
    mkdir -p "${capture_root}"
    "${adb_exe}" shell \
        "mkdir -p ${remote_root}/replay_capture && rm -f ${remote_root}/replay_capture/*"
    capture_env="QBH_REPLAY_DUMP_DIR=${remote_root}/replay_capture"
fi
set +e
"${adb_exe}" shell \
    "cd ${remote_root} && ${runtime_env} ${capture_env} QBH_VERTICAL_SLICE=1 QBH_REPLAY_SEQUENCE=1 QBH_SCAN_MODE=prefill QBH_LOGICAL_M=64 QBH_KV_CACHE_LENGTH=0 QBH_KV_CACHE_CAPACITY=72 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} 1 ${runtime_args}"
run_status=$?
set -e
if [[ -n "${capture_root}" ]]; then
    "${adb_exe}" pull "${remote_root}/replay_capture/." \
        "$(wslpath -w "${capture_root}")" >/dev/null
fi
exit "${run_status}"
