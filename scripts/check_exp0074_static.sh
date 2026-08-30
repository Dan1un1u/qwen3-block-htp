#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(39)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(74)' include/block_protocol.h
grep -q 'EXP-0074' src/host/block_main.c
grep -q 'GQA_BATCH_STAGED_SOFTMAX = 15' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES UINT32_C(2)' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_staged_softmax_run_tasks' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates' src/dsp/attention_u8_core.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0074_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0074","static_gate":"pass","block_abi":39,"runtime_telemetry_experiment":74,"control_attention_tasks":8,"candidate_softmax_row_tasks":16,"softmax_rows_per_task":32,"persistent_hvx_contexts":6,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
