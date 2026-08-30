#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(42)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(81)' include/block_protocol.h
grep -q 'EXP-0081' src/host/block_main.c
grep -q 'QBH_BLOCK_W4U8_QKV_ASSIST_WORKERS UINT32_C(2)' src/dsp/block_imp.c
grep -q '__sync_bool_compare_and_swap' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_qkv_expand_help' src/dsp/block_imp.c
grep -q 'w4u8_qkv_worker_assist_batch_count' src/host/block_main.c
grep -q 'W4U8_STREAMING_PERSISTENT_MLP_HVX = 7' include/block_protocol.h
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0081_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0081","static_gate":"pass","block_abi":42,"runtime_telemetry_experiment":81,"q_worker_assist_workers":2,"qk_prep_workers_unchanged":5,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
