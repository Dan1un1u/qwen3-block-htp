#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(38)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(67)' include/block_protocol.h
grep -q 'INPUT_RSQRT = 4' include/block_protocol.h
grep -q 'qbh_hvx_rms_norm_u8_native_activation_batched_rsqrt' \
    include/hvx_u8_ops.h src/dsp/hvx_u8_ops.c src/dsp/block_imp.c
grep -q 'u8_input_norm_actual_hash' \
    include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'qhmath_hvx_rsqrt_vf' src/dsp/hvx_u8_ops.c
grep -q 'FUSED_QK_REQUANT_HMX_BATCH_LUT_TEMPLATES = 13' \
    include/block_protocol.h
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0067_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0067","static_gate":"pass","block_abi":38,"runtime_telemetry_experiment":67,"control":"scalar-input-rsqrt-per-row","candidate":"two-vector-input-rsqrt","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
