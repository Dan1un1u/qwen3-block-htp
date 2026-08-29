#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
serial="${ANDROID_SERIAL:-3B15C8007Z300000}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0045-block}"
package_dir="${QBH_EXP0045_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_head="$(git -C "${project_root}" rev-parse --short=12 HEAD)"
result_dir="${1:-/mnt/d/llm_exp/results/qwen3-block-htp/exp0045/${stamp}_${source_head}_attention_audit}"
mode="${2:-qkv_batch2}"
dump_dir="${result_dir}/attention_dump"
remote_dump="${remote_root}/attention-audit-${stamp}"

case "${mode}" in
    qkv_batch2|qkv_batch4|qkvo_batch4)
        ;;
    *)
        printf 'usage: %s [result-dir] [qkv_batch2|qkv_batch4|qkvo_batch4]\n' "$0" >&2
        exit 2
        ;;
esac

mkdir -p "${dump_dir}"
"${adb_exe}" -s "${serial}" get-state >/dev/null
QBH_DUMP_ATTENTION_DIR="${remote_dump}" \
    "${adb_exe}" -s "${serial}" shell \
    "mkdir -p ${remote_dump} && cd ${remote_root} && QBH_DUMP_ATTENTION_DIR=${remote_dump} LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 W4U8 1 2 32 rms_rope_softmax on on fused serial control hvx w4u8_streaming 3 64 u8_log2_gqa_qkv_overlap 4 control ${mode}" \
    | tee "${result_dir}/device_audit.json"
"${adb_exe}" -s "${serial}" pull "${remote_dump}/." \
    "$(wslpath -w "${dump_dir}")" >/dev/null

/home/daniuniu/mllm-quant-venv/bin/python \
    "${project_root}/scripts/verify_exp0042_attention_dump.py" \
    --experiment EXP-0045 \
    --dump "${dump_dir}" \
    --package "${package_dir}" \
    --output "${result_dir}/implementation_reference.json" \
    | tee "${result_dir}/implementation_reference.stdout.json"

"${adb_exe}" -s "${serial}" shell \
    "rm -f ${remote_dump}/actual_q_tiles_u8.bin ${remote_dump}/actual_k_tiles_u8.bin ${remote_dump}/actual_v_tiles_u8.bin ${remote_dump}/actual_score_tiles_u8.bin ${remote_dump}/actual_probability_tiles_u8.bin ${remote_dump}/actual_av_tiles_u8.bin && rmdir ${remote_dump}" >/dev/null

printf 'EXP0045_AUDIT_RESULT=%s\n' "${result_dir}"
