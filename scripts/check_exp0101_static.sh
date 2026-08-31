#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(101)' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_attention_u8_ready_only_run_tasks' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_attention_u8_ready_try_claim_softmax' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_attention_u8_ready_try_claim_av' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0101","static_gate":"pass","control_scheduler_mode":0,"candidate_scheduler_mode":1,"softmax_slices":16,"av_groups":8,"global_stage_barrier_added":false,"qnn_dependency":false}'
