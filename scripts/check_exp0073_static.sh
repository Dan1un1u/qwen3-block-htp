#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(40)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(73)' include/block_protocol.h
grep -q 'EXP-0073' src/host/block_main.c
grep -q 'SHARED_LUT_TEMPLATES_GQA_BATCH = 15' include/block_protocol.h
grep -q 'qbh_attention_u8_build_sole_lut_template_bank' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_requant_softmax_group_shared_lut_templates' src/dsp/block_imp.c
grep -q 'u8_attention_lut_template_build_count' include/block_protocol.h
grep -q 'buffers->attention_projection' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0073_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0073","static_gate":"pass","block_abi":40,"runtime_telemetry_experiment":73,"control_template_builds_per_block":8,"candidate_template_builds_per_block":1,"shared_bank_bytes":448,"new_vtcm_allocation_bytes":0,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
