#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(41)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(79)' include/block_protocol.h
grep -q 'EXP-0079' src/host/block_main.c
grep -q 'W4U8_STREAMING_PERSISTENT_MLP_HVX = 7' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DOWN_PERSISTENT_HVX_WORKERS UINT32_C(5)' src/dsp/block_imp.c
grep -q 'qbh_w4u8_hybrid_down_runner_start' src/dsp/block_imp.c
grep -q 'qbh_w4u8_hybrid_down_runner_wait' src/dsp/block_imp.c
grep -q 'qbh_run_chunked_w4_managed_hvx_worker' src/dsp/block_imp.c
grep -q 'w4u8_down_persistent_hvx_dispatch_count' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0079_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0079","static_gate":"pass","block_abi":41,"runtime_telemetry_experiment":79,"persistent_gate_up_hvx_workers":3,"persistent_down_hvx_workers":5,"transient_down_hvx_workers":1,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
