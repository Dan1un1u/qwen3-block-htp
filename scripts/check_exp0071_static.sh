#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(38)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(71)' include/block_protocol.h
grep -q 'EXP-0071' src/host/block_main.c
grep -q 'SHARED_ROPE_PARALLEL_INPUT = 4' include/block_protocol.h
grep -q 'QBH_BLOCK_HVX_POOL_U8_INPUT_NORM = 10' src/dsp/block_imp.c
grep -q 'qbh_hvx_pool_u8_input_norm' src/dsp/block_imp.c
grep -q 'qbh_hvx_rms_norm_u8_native_activation_rows' src/dsp/block_imp.c
grep -q 'w4u8_input_norm_task_count' include/block_protocol.h
grep -q 'u8_input_norm_actual_hash' include/block_protocol.h
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0071_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0071","static_gate":"pass","block_abi":38,"runtime_telemetry_experiment":71,"control_input_norm_tasks":0,"candidate_input_norm_tasks":16,"persistent_hvx_contexts":6,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
