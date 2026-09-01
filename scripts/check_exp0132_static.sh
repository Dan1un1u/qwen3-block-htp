#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(52)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(132)' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_atomic_fetch_add_two' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'claimed_regions \* pool->region_tiles' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0132 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0132","static_gate":"pass","recipe":"W4F16","control_claim_regions":1,"candidate_claim_regions":2,"region_tiles":32,"pool_workers":2,"main_hvx_contexts":1,"hmx_batch_outputs":8,"one_hmx_owner":true,"qnn_dependency":false}'
