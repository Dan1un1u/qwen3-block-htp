#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(30)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(57)' include/block_protocol.h
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER = 9' include/block_protocol.h
grep -q 'qbh_attention_u8_pack_v_native_vgather' src/dsp/attention_u8_core.c
grep -q 'Q6_vgather_AQRMVh' src/dsp/attention_u8_core.c
grep -q 'u8_log2_gqa_qkv_overlap_vgather' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0057_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0057","static_gate":"pass","block_abi":30,"runtime_telemetry_experiment":57,"control":"scalar-v-recenter-lut","candidate":"two-half-vector-hvx-vgather","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
