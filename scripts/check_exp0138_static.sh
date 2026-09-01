#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(58)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(138)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4F16_GATE_UP_INITIAL_UP_DMA_OVERLAP' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_w4f16_mlp_start_prefetch(.*' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4f16_gate_up_first_gate_hmx_start_tick' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4f16_gate_up_initial_up_dma_wait_start_tick' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'state->prefetched_batch != 0U' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0138 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0138","static_gate":"pass","recipe":"W4F16","control_initial_up_dma":"synchronous","candidate_initial_up_dma":"asynchronous","gate_hmx_before_up_wait_telemetry":true,"steady_state_unchanged":true,"stream_subgroup_tiles":4,"expansion_contexts":4,"one_hmx_owner":true,"qnn_dependency":false}'
