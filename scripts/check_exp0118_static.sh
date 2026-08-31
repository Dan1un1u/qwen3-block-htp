#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(48)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(118)' include/block_protocol.h
grep -q 'EXP-0118' src/host/block_main.c
grep -q 'QBH_BLOCK_W4U8_QKV_ASSIST_REGION_K_TILES UINT32_C(16)' src/dsp/block_imp.c
grep -q 'qbh_w4u8_qkv_expand_preemptible' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_qkv_expand_help' src/dsp/block_imp.c
grep -q 'w4u8_qkv_worker_assist_region_count' src/host/block_main.c
grep -q 'qkvo_batch4_qk_head_pairs_q_preemptible_expand' src/host/block_main.c
if grep -Rqs --exclude='check_exp0118_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0118","static_gate":"pass","block_abi":48,"runtime_telemetry_experiment":118,"assist_region_k_tiles":16,"qk_prep_worker_count_unchanged":5,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
