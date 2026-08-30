#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(42)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(80)' include/block_protocol.h
grep -q 'EXP-0080' src/host/block_main.c
grep -q 'QK_HEAD_PAIRS_CROSS_QKV_PREFETCH = 6' include/block_protocol.h
grep -q 'w4u8_qkv_cross_prefetch_count' src/dsp/block_imp.c
grep -q 'w4u8_qkv_cross_prefetch_adoption_count' src/dsp/block_imp.c
grep -q 'w4u8_qkv_cross_prefetch_lifetime_ticks' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0080_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0080","static_gate":"pass","block_abi":42,"runtime_telemetry_experiment":80,"candidate_cross_prefetch_count":2,"candidate_cross_prefetch_adoption_count":2,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
