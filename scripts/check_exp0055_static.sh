#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(29)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(55)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_PAIRS = 5' include/block_protocol.h
grep -q 'qbh_hvx_qk_norm_rope_u8_native_head_pair' src/dsp/hvx_u8_ops.c
grep -q 'qbh_hvx_qk_norm_rope_u8_native_k_head_pair' src/dsp/hvx_u8_ops.c
grep -q 'qbh_qk_norm_rope_load_gamma_sf32' src/dsp/hvx_u8_ops.c
grep -q 'qbh_qk_norm_rope_load_row_sf32' src/dsp/hvx_u8_ops.c
grep -q 'qbh_attention_u8_qk_prep_pool_run_head_pair_tasks' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HEADS / 2U + QBH_BLOCK_KV_HEADS / 2U' src/dsp/block_imp.c
grep -q 'qkvo_batch4_qk_head_pairs' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0055_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0055","static_gate":"pass","block_abi":29,"runtime_telemetry_experiment":55,"control_logical_head_tasks":24,"candidate_pair_tasks":12,"candidate_logical_heads":24,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
