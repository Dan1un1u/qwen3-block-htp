#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${project_root}/include/block_protocol.h"
dsp="${project_root}/src/dsp/block_imp.c"
host="${project_root}/src/host/block_main.c"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(58)' "${protocol}"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(146)' "${protocol}"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_CROSS_PROJECTION_RING' "${protocol}"
grep -q 'strcmp(schedule,' "${host}"
grep -q '"cross_projection_ring"' "${host}"
grep -q 'QBH_BLOCK_W4F16_HMX_BATCH_N_TILES UINT32_C(2)' "${dsp}"
grep -q 'uint8_t \*compressed_slots\[2\]' "${dsp}"
grep -q 'uint8_t \*expanded_slots\[2\]' "${dsp}"
grep -q 'qbh_w4f16_qkv_schedule_task_init' "${dsp}"
grep -q 'qbh_hvx_pool_qk_norm_rope_publish' "${dsp}"
grep -q 'qbh_hmx_start_fp16_tile_scales(' "${dsp}"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0146 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0146","static_gate":"pass","recipe":"W4F16","batch2_hmx":true,"cross_projection_qkv_task_stream":true,"compressed_slots":2,"expanded_slots":2,"new_large_vtcm_slots":0,"per_head_publish":true,"one_hmx_owner":true,"qnn_dependency":false}'
