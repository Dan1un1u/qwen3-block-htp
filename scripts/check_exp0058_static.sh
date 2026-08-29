#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(31)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(58)' include/block_protocol.h
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL = 10' include/block_protocol.h
grep -q 'qbh_attention_u8_pack_v_native_vgather_vdeal' src/dsp/attention_u8_core.c
test "$(grep -c 'recentered = Q6_Vb_vdeal_Vb(recentered)' src/dsp/attention_u8_core.c)" -eq 5
grep -q 'u8_log2_gqa_qkv_overlap_vgather_vdeal' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0058_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0058","static_gate":"pass","block_abi":31,"runtime_telemetry_experiment":58,"control":"vgather-scalar-transpose","candidate":"vgather-five-vdeal-transpose","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
