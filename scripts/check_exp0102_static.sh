#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(102)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_MAX_BATCH_N_TILES UINT32_C(8)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'header->w4u8_qkv_batch_n_tiles_config' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'for (uint32_t head = first_head; head < end_head; ++head)' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0102","static_gate":"pass","control_qkv_batch_n_tiles":4,"candidate_qkv_batch_n_tiles":8,"o_batch_n_tiles":4,"control_qkv_commands":32,"candidate_qkv_commands":16,"control_total_hmx_commands":176,"candidate_total_hmx_commands":160,"qnn_dependency":false}'
