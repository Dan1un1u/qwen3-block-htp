#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(37)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(65)' include/block_protocol.h
grep -q 'FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES = 13' include/block_protocol.h
grep -q 'qbh_attention_u8_build_sole_lut_templates' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_requant_softmax_group_lut_templates' src/dsp/block_imp.c
grep -q 'QBH_ATTN_U8_SOFTMAX_SCRATCH_BYTES UINT32_C(768)' include/attention_u8_core.h
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0065_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0065","static_gate":"pass","block_abi":37,"runtime_telemetry_experiment":65,"control":"per-row-scalar-sole-lut","candidate":"per-group-precomputed-sole-lut-templates","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
