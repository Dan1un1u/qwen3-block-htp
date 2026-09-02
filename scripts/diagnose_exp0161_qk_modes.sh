#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
remote_root="/data/local/tmp/qwen3-block-htp/exp0161-l64-segmented"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0161/diagnostic_qk_modes}"
model_root="/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/decode_l64_w4u8_hmx_segmented_v4c_exact"

mkdir -p "${result_dir}"
for mode in 0 1 2; do
    remote_capture="${remote_root}/qkmode_${mode}"
    "${adb_exe}" shell \
        "mkdir -p ${remote_capture} && rm -f ${remote_capture}/block_output_u8.bin ${remote_capture}/actual_kv_cache_k_u8.bin ${remote_capture}/actual_kv_cache_v_u8.bin" \
        >/dev/null
    set +e
    "${adb_exe}" shell \
        "cd ${remote_root} && QBH_DUMP_CACHE_DIR=${remote_capture} QBH_DUMP_OUTPUT_PATH=${remote_capture}/block_output_u8.bin QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8_segmented_v4 QBH_W4U8_DELTA_RECONSTRUCTION=serial QBH_SCAN_MODE=decode QBH_LOGICAL_M=1 QBH_KV_CACHE_LENGTH=64 QBH_KV_CACHE_CAPACITY=65 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 ${mode}" \
        > "${result_dir}/mode_${mode}.jsonl" \
        2> "${result_dir}/mode_${mode}.stderr"
    runner_status=$?
    set -e
    printf '%s\n' "${runner_status}" \
        > "${result_dir}/mode_${mode}.runner_status"
    for file in block_output_u8.bin actual_kv_cache_k_u8.bin actual_kv_cache_v_u8.bin; do
        "${adb_exe}" pull "${remote_capture}/${file}" \
            "$(wslpath -w "${result_dir}/mode_${mode}_${file}")" >/dev/null
    done
done

norm_modes=(
    scalar
    hvx_tree
    hvx_tree_qk_batched_rsqrt
    hvx_tree_qk_batched_rsqrt_shared_rope
    hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input
)
for norm_index in 0 1 2 3 4; do
    norm_mode="${norm_modes[${norm_index}]}"
    remote_capture="${remote_root}/normmode_${norm_index}"
    "${adb_exe}" shell \
        "mkdir -p ${remote_capture} && rm -f ${remote_capture}/block_output_u8.bin ${remote_capture}/actual_kv_cache_k_u8.bin ${remote_capture}/actual_kv_cache_v_u8.bin" \
        >/dev/null
    set +e
    "${adb_exe}" shell \
        "cd ${remote_root} && QBH_DUMP_CACHE_DIR=${remote_capture} QBH_DUMP_OUTPUT_PATH=${remote_capture}/block_output_u8.bin QBH_W4U8_STREAM_FENCE=single_fence QBH_W4U8_GATE_UP_RING_SLOTS=16 QBH_W4U8_QKV_RING_EXPAND_WORKERS=3 QBH_KV_CACHE_LAYOUT=hmx_native_u8_segmented_v4 QBH_W4U8_DELTA_RECONSTRUCTION=serial QBH_SCAN_MODE=decode QBH_LOGICAL_M=1 QBH_KV_CACHE_LENGTH=64 QBH_KV_CACHE_CAPACITY=65 LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs ${norm_mode} control 4 4 4 0" \
        > "${result_dir}/norm_${norm_index}.jsonl" \
        2> "${result_dir}/norm_${norm_index}.stderr"
    runner_status=$?
    set -e
    printf '%s\n' "${runner_status}" \
        > "${result_dir}/norm_${norm_index}.runner_status"
    for file in block_output_u8.bin actual_kv_cache_k_u8.bin actual_kv_cache_v_u8.bin; do
        "${adb_exe}" pull "${remote_capture}/${file}" \
            "$(wslpath -w "${result_dir}/norm_${norm_index}_${file}")" >/dev/null
    done
done

"${python_exe}" - "${result_dir}" "${model_root}" <<'PY'
import json
import pathlib
import sys

import numpy as np

result = pathlib.Path(sys.argv[1])
model = pathlib.Path(sys.argv[2])
summary = {}
for family, count in (("mode", 3), ("norm", 5)):
    for mode in range(count):
        entry = {
            "runner_status": int(
                (result / f"{family}_{mode}.runner_status").read_text().strip()
            )
        }
        for kind in ("k", "v"):
            actual = np.fromfile(
                result / f"{family}_{mode}_actual_kv_cache_{kind}_u8.bin",
                dtype=np.uint8,
            )
            reference = np.fromfile(
                model / f"reference_kv_cache_{kind}_u8.bin",
                dtype=np.uint8,
            ).reshape(8, 65, 128)[:, 64]
            head_bytes = actual.size // 8
            row_offset = (
                2 * (4096 + 256) if kind == "k" else 2 * 4096 + 1024
            )
            current = np.stack([
                actual[head * head_bytes + row_offset:
                       head * head_bytes + row_offset + 128]
                for head in range(8)
            ])
            difference = np.abs(
                current.astype(np.int16) - reference.astype(np.int16)
            )
            entry[f"{kind}_mismatches"] = int(np.count_nonzero(difference))
            entry[f"{kind}_max_lsb"] = int(difference.max(initial=0))
        summary[f"{family}_{mode}"] = entry
(result / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
