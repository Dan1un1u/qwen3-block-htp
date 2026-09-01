#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(51)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(139)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4U8_QKV_MAIN_PREP_ASSIST' \
    "${project_root}/src/host/block_main.c"
grep -q 'main_prep_job.worker_index = pool->worker_count' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4u8_qkv_dma_feed_complete_tick = assist_start' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4u8_qkv_main_hmx_wait_start_tick = wait_start' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'attention_qk_main_completed_groups' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0139 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0139","static_gate":"pass","recipe":"W4U8","control_expand_workers":3,"control_prep_workers":2,"candidate_expand_workers":3,"candidate_prep_workers":2,"main_assist_after_dma_feed":true,"main_assist_before_hmx_wait":true,"one_hmx_owner":true,"qnn_dependency":false}'
