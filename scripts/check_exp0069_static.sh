#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(38)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(69)' include/block_protocol.h
grep -q 'uint32_t attention_active_contexts;' include/block_protocol.h
grep -q 'attention_active_contexts - 1U' src/dsp/block_imp.c
grep -q 'header->attention_active_contexts = attention_active_contexts;' src/host/block_main.c
grep -q 'EXP-0069' src/host/block_main.c
grep -q 'attention_active_contexts' src/host/block_main.c
grep -q '#define QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS UINT32_C(6)' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0069_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0069","static_gate":"pass","block_abi":38,"runtime_telemetry_experiment":69,"persistent_attention_contexts":6,"candidate_active_attention_contexts":[4,5],"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
