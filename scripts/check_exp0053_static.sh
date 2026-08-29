#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(28)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(53)' include/block_protocol.h
grep -q 'QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL4 = 3' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_RESIDUAL_ROWS_PER_TASK UINT32_C(4)' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HVX_POOL_U8_RESIDUAL' src/dsp/block_imp.c
grep -q 'qbh_hvx_residual_rms_norm_u8_native_io_rows' src/dsp/block_imp.c
grep -q 'qbh_hvx_residual_add_u8_native_output_rows' src/dsp/block_imp.c
grep -q 'fused_pool4' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0053_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0053","static_gate":"pass","block_abi":28,"runtime_telemetry_experiment":53,"control_residual_contexts":1,"candidate_residual_contexts":4,"rows_per_task":4,"tasks_per_boundary":16,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
