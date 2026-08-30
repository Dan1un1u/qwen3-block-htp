#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(86)' include/block_protocol.h
grep -q 'QBH_BLOCK_DOWN_TIMELINE_ENABLED UINT32_C(1)' include/block_protocol.h
grep -q 'qbh_projection_layout_set_two_chunk_split' include/hmx_u8s8_projection.h
grep -q 'first_continuation_ready_tick' include/w4_parallel_pipeline.h
grep -q 'w4u8_down_first_chunk_tiles_requested' src/host/block_main.c
grep -q 'w4u8_down_continuation_ready_wait_max_ticks' src/host/block_main.c
if git grep -n 'QnnGraph\|QnnContext\|QnnBackend' -- include src >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0086","static_gate":"pass","qnn_dependency":false}\n'
