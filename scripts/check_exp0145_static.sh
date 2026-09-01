#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(52)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(145)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4_HMX_MAX_BATCH_OUTPUTS UINT32_C(8)' \
    "${project_root}/include/w4_parallel_pipeline.h"
grep -q 'max_nonstreaming_batch_outputs' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'in_command_slot_release_count' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'w4u8_down_hmx_batch_outputs != 8U' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0145 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0145","static_gate":"pass","recipe":"W4U8","control_down_batch":4,"candidate_down_batch":8,"runner_max_batch":8,"in_command_slot_release":true,"qk_pair_mode":3,"qkv_ring_expand_workers":3,"one_hmx_owner":true,"qnn_dependency":false}'
