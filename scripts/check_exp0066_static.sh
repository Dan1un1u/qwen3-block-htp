#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(38)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(66)' include/block_protocol.h
grep -q 'LUT_TEMPLATES_AUDIT_REDUCTIONS = 14' include/block_protocol.h
grep -q 'audit_reductions_only == 0U || telemetry != NULL' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_requant_softmax_group_audit_reductions' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0066_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0066","static_gate":"pass","block_abi":38,"runtime_telemetry_experiment":66,"control":"unconditional-probability-row-sum-reduction","candidate":"audit-only-probability-row-sum-reduction","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
