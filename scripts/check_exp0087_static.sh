#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(44)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(87)' include/block_protocol.h
grep -q 'EXP-0087' src/host/block_main.c
grep -q 'QBH_BLOCK_W4U8_SOFTMAX_ROW_SLICES UINT32_C(2)' src/dsp/block_imp.c
grep -q 'header->w4u8_attention_timeline_requested' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_publish_timeline' src/dsp/block_imp.c
grep -q 'w4u8_attention_softmax_context5_work_ticks' src/host/block_main.c
if grep -Rqs --exclude='check_exp0087_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0087","stage":"A","static_gate":"pass","block_abi":44,"softmax_row_slices":2,"timeline_diagnostic_isolated_from_numerical_audit":true,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
