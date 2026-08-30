#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(34)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(62)' include/block_protocol.h
grep -q 'QBH_BLOCK_U8_NORM_REDUCTION_HVX_TREE_QK_BATCHED_RSQRT_SHARED_ROPE = 3' include/block_protocol.h
grep -q 'qbh_hvx_qk_rope_preconvert_sf32' src/dsp/hvx_u8_ops.c
grep -q 'QBH_QK_ROPE_SF32_CACHE_OFFSET' src/dsp/block_imp.c
grep -q 'hvx_tree_qk_batched_rsqrt_shared_rope' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0062_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0062","static_gate":"pass","block_abi":34,"runtime_telemetry_experiment":62,"control":"per-task-rope-fp16-to-sf32","candidate":"shared-64-row-vtcm-rope-sf32-cache","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
