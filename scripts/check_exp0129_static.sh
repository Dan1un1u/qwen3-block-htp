#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(51)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(129)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_MAX_REQUESTED_HVX_WORKERS UINT32_C(4)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4f16_requested_hvx_workers != 4U' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'w4f16_hvx_workers != 4U' \
    "${project_root}/src/host/block_main.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0129 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0129","static_gate":"pass","recipe":"W4F16","control_workers":3,"candidate_workers":4,"region_tiles":32,"hmx_batch_outputs":8,"one_hmx_owner":true,"qnn_dependency":false}'
