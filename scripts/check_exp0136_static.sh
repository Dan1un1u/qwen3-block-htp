#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(56)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(136)' \
    "${project_root}/include/block_protocol.h"
grep -q 'w4f16_gate_up_stream_group_tiles' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_hvx_silu_multiply_f16_crouton_tile_range' \
    "${project_root}/src/dsp/hvx_fp16_ops.c"
grep -q 'first_source_tile + column_tile' \
    "${project_root}/src/dsp/hvx_fp16_ops.c"
grep -q 'mlp_crouton_subgroup_done' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'subgroups_per_hmx' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4F16_GATE_UP_STREAM_GROUP_TILES' \
    "${project_root}/src/host/block_main.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0136 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0136","static_gate":"pass","recipe":"W4F16","control_stream_group_tiles":8,"candidate_stream_group_tiles":4,"hmx_group_tiles":8,"candidate_subgroups_per_hmx":2,"candidate_tasks_per_block":48,"final_stream_contexts":5,"one_hmx_owner":true,"qnn_dependency":false}'
