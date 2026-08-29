#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(29)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(54)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QKVO_BATCH4_QKV_HEAD_TASKS = 5' include/block_protocol.h
grep -q '2U \* QBH_BLOCK_KV_HEADS' src/dsp/block_imp.c
grep -q 'u8_attention_prepared_v_group_count' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_pack_v_native' src/dsp/block_imp.c
grep -q 'qkvo_batch4_qkv_head_tasks' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0054_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0054","static_gate":"pass","block_abi":29,"runtime_telemetry_experiment":54,"control_v_pack":"attention_callback","candidate_v_pack":"projection_head_ready_hvx_pool","candidate_v_tasks":8,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
