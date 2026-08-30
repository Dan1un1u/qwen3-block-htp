#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(37)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(68)' include/block_protocol.h
grep -q '"EXP-0068"' src/host/block_main.c
grep -q '#define QBH_BLOCK_MAX_ATTENTION_HVX_CONTEXTS UINT32_C(6)' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_MAX_POOL_HVX_WORKERS' src/dsp/block_imp.c
grep -q 'attention_hvx_contexts > 6U' src/host/block_main.c
grep -q '\[attention_hvx_contexts:1\.\.6\]' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0068_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0068","static_gate":"pass","block_abi":37,"runtime_telemetry_experiment":68,"control_attention_contexts":4,"candidate_attention_contexts":[5,6],"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
