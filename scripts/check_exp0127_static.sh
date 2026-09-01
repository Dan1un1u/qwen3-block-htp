#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(50)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(127)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_V_BATCH4' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_KV_BATCH4' \
    "${project_root}/include/block_protocol.h"
grep -q 'v_batch4' "${project_root}/src/host/block_main.c"
grep -q 'kv_batch4' "${project_root}/src/host/block_main.c"
grep -q 'qbh_w4f16_projection_group_tiles' \
    "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0127 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0127","static_gate":"pass","recipe":"W4F16","control_batch_tiles":2,"v_batch_tiles":4,"kv_batch_tiles":4,"q_batch_tiles":2,"one_hmx_owner":true,"qnn_dependency":false}'
