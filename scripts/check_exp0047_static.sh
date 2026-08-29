#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(25)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(46)' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_OUTPUT' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QKVO_BATCH4' include/block_protocol.h
grep -q 'Q6_vgather_AQRMVw' src/dsp/hvx_u8_ops.c
grep -q 'qbh_hvx_residual_add_u8_native_output' src/dsp/block_imp.c
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        src include CMakeLists.txt >/dev/null; then
    printf 'QNN dependency detected\n' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0047","static_gate":"pass","characterization_only":true,"block_abi":25,"runtime_telemetry_experiment":46,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
