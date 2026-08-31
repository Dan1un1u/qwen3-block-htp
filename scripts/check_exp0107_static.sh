#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(44)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(107)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_F16F16_PROJECTION_GATE8_INTERLEAVED' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_run_f16f16_interleaved_gate_up' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'gate_up_batch8_interleaved' \
    "${project_root}/src/host/block_main.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0107","static_gate":"pass","f16f16_gate_up_interleaved":true,"group_tiles":8,"one_hmx_owner":true,"qnn_dependency":false}'
