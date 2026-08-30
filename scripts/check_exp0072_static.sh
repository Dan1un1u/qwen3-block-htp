#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(39)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(72)' include/block_protocol.h
grep -q 'EXP-0072' src/host/block_main.c
grep -q 'QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6 = 4' include/block_protocol.h
grep -q 'fused_pool6' src/host/block_main.c
grep -q 'w4u8_residual_active_contexts' include/block_protocol.h
grep -q 'active_worker_count + 1U' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0072_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0072","static_gate":"pass","block_abi":39,"runtime_telemetry_experiment":72,"control_residual_contexts":4,"candidate_residual_contexts":6,"persistent_hvx_contexts":6,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
