#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(40)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(78)' include/block_protocol.h
grep -q 'EXP-0078' src/host/block_main.c
grep -q 'W4U8_STREAMING_PERSISTENT_GATE_UP_HVX = 6' include/block_protocol.h
grep -q 'qbh_run_chunked_w4_pipeline_external_hvx' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HVX_POOL_W4U8_PIPELINE' src/dsp/block_imp.c
grep -q 'qbh_w4u8_pipeline_pool_start' src/dsp/block_imp.c
grep -q 'qbh_w4u8_pipeline_pool_wait' src/dsp/block_imp.c
grep -q 'w4u8_gate_up_persistent_hvx_dispatch_count' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0078_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0078","static_gate":"pass","block_abi":40,"runtime_telemetry_experiment":78,"persistent_gate_up_hvx_workers":3,"transient_down_hvx_workers":6,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
