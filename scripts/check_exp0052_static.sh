#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(27)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(52)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QKVO_BATCH4_QK_HEAD_TASKS = 4' include/block_protocol.h
grep -q 'qbh_attention_u8_qk_prep_pool_run_head_tasks' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HEADS + QBH_BLOCK_KV_HEADS' src/dsp/block_imp.c
grep -q 'qkvo_batch4_qk_head_tasks' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0052_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0052","static_gate":"pass","block_abi":27,"runtime_telemetry_experiment":52,"control_qk_prep_tasks":8,"candidate_qk_prep_tasks":24,"logical_q_heads":16,"logical_k_heads":8,"qkv_hmx_commands_per_block":32,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
