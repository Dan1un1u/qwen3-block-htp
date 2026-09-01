#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(55)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(135)' \
    "${project_root}/include/block_protocol.h"
grep -q 'w4f16_gate_up_extra_stream_worker' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'pool->jobs\[extra_stream_worker\].command_kind' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'command_ready\[extra_stream_worker\]' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'extra_stream_worker != UINT32_MAX ? 4U : 3U' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0135 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0135","static_gate":"pass","recipe":"W4F16","expansion_contexts":4,"control_final_stream_contexts":4,"candidate_final_stream_contexts":5,"candidate_pool_workers":[0,1,2,3],"candidate_main_context":true,"region_tiles":32,"claim_regions":1,"hmx_batch_outputs":8,"one_hmx_owner":true,"qnn_dependency":false}'
