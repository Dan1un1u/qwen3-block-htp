#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(59)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(152)' include/block_protocol.h
grep -q 'QBH_VERTICAL_SLICE_FIRST_LAYER UINT32_C(0)' include/block_protocol.h
grep -q 'QBH_VERTICAL_SLICE_LAYER_COUNT QBH_QWEN3_TRANSFORMER_LAYERS' \
    include/block_protocol.h
grep -q '#pragma weak rpcmem_alloc2' src/host/block_main.c
grep -q 'shared = rpcmem_alloc2' src/host/block_main.c
grep -q 'QBH_LAYOUT_ONLY' src/host/block_main.c
grep -q 'QBH_BLOCK_FULL_STACK_HIDDEN_CAPTURE' include/block_protocol.h
grep -q 'QBH_HIDDEN_CAPTURE_DIR' src/host/block_main.c
grep -q 'full_stack_hidden_capture_ddr_write_bytes' src/dsp/block_imp.c
grep -q 'QBH_REPLAY_FP16_MAX_COMPOSED_NRMSE (0.003)' \
    src/host/block_main.c
grep -q 'QBH_REPLAY_FP16_MAX_CACHE_VIOLATION_FRACTION (0.01)' \
    src/host/block_main.c
grep -q 'fp16_gate_version.*composition_v2' src/host/block_main.c
grep -q 'cache_structure_mismatches' src/host/block_main.c
grep -q 'cache_composed_cosine_diagnostic_failure_count' \
    src/host/block_main.c
grep -q 'QBH_W4U8_BOUNDARY_AUDIT' src/host/block_main.c
grep -q 'QBH_DUMP_BOUNDARY_DIR' src/host/block_main.c
grep -q 'w4u8_boundary_audit_ddr_write_bytes += boundary_bytes' \
    src/dsp/block_imp.c
if grep -q 'total_bytes > INT_MAX' src/host/block_main.c; then
    printf 'legacy INT_MAX allocation guard remains\n' >&2
    exit 1
fi
/home/daniuniu/.cache/qwen3-block-htp-py/bin/python -m py_compile \
    scripts/export_exp0149_vertical_slice.py \
    scripts/audit_exp0152_hidden_trajectory.py \
    scripts/audit_exp0152_replay_composition.py
printf 'EXP-0152 static gate passed\n'
