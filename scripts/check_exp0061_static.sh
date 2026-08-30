#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(33)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(61)' include/block_protocol.h
grep -q 'QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT = 2' include/block_protocol.h
grep -q 'qbh_qk_norm_rope_pair_batched_rsqrt' src/dsp/hvx_u8_ops.c
grep -q 'qhmath_hvx_rsqrt_vf' src/dsp/hvx_u8_ops.c
grep -q 'hvx_tree_qk_batched_rsqrt' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0061_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0061","static_gate":"pass","block_abi":33,"runtime_telemetry_experiment":61,"control":"exact-hvx-word-tree-plus-scalar-qk-rsqrt","candidate":"exact-hvx-word-tree-plus-four-vector-qk-rsqrt","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
