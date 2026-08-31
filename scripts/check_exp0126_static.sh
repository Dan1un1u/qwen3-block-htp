#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(49)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(126)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_HEAD_ALIGNED_BATCH4' \
    "${project_root}/include/block_protocol.h"
grep -q 'head_aligned_batch4' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_w4f16_projection_group_tiles' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_HEAD_DIM / QBH_HMX_FP16_COLS' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0126 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0126","static_gate":"pass","recipe":"W4F16","control_batch_tiles":2,"candidate_batch_tiles":4,"head_channels":128,"one_hmx_owner":true,"qnn_dependency":false}'
