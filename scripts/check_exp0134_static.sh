#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(54)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(134)' \
    "${project_root}/include/block_protocol.h"
grep -q 'w4f16_gate_up_extra_expand_worker' \
    "${project_root}/src/dsp/block_imp.c"
grep -q '? 3U : UINT32_MAX' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'pool->jobs\[pool->extra_expand_worker_index\].command_kind' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'command_done\[pool->extra_expand_worker_index\]' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'first_worker = 2U' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0134 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0134","static_gate":"pass","recipe":"W4F16","control_expansion_contexts":3,"candidate_expansion_contexts":4,"candidate_pool_workers":[0,1,3],"reserved_stream_worker":2,"region_tiles":32,"claim_regions":1,"hmx_batch_outputs":8,"one_hmx_owner":true,"qnn_dependency":false}'
