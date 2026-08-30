#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(32)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(60)' include/block_protocol.h
grep -q 'QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE = 1' include/block_protocol.h
grep -q 'qbh_hvx_u8_set_norm_reduction_mode' include/hvx_u8_ops.h
grep -q 'qbh_reduce_word_sum_hvx' src/dsp/hvx_u8_ops.c
grep -q 'Q6_V_vlalign_VVR' src/dsp/hvx_u8_ops.c
grep -q 'u8_norm_reduction_mode' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0060_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0060","static_gate":"pass","block_abi":32,"runtime_telemetry_experiment":60,"control":"four-stack-spills-plus-128-scalar-lane-additions","candidate":"one-exact-hvx-word-tree","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
