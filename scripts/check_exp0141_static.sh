#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(50)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(141)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_BATCH_N_TILES UINT32_C(4)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_W4U8_O_MAX_BATCH_N_TILES UINT32_C(8)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'return header->w4u8_o_batch_n_tiles' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4u8_o_batch_n_tiles_observed = batch_tiles' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4U8_O_BATCH_N_TILES' \
    "${project_root}/src/host/block_main.c"
grep -q 'w4u8_qkv_ring_expand_workers' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0141 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0141","static_gate":"pass","recipe":"W4U8","control_o_batch_n_tiles":4,"candidate_o_batch_n_tiles":8,"qkv_batch_n_tiles":4,"control_o_commands_per_block":16,"candidate_o_commands_per_block":8,"qkv_ring_expand_workers":3,"gate_up_ring_slots":16,"same_binary_and_buffer_plan":true,"one_hmx_owner":true,"qnn_dependency":false}'
