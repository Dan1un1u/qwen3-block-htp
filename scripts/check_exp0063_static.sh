#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(35)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(63)' include/block_protocol.h
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_FUSED_QK_REQUANT = 11' include/block_protocol.h
grep -q 'qbh_attention_u8_requant_softmax_group' include/attention_u8_core.h
grep -q 'qbh_attention_u8_fused_qk_requant_enabled' src/dsp/block_imp.c
grep -q 'u8_attention_qk_requant_softmax_ticks' src/host/block_main.c
grep -q 'hvx_tree_qk_batched_rsqrt_shared_rope' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0063_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0063","static_gate":"pass","block_abi":35,"runtime_telemetry_experiment":63,"control":"separate-qk-requant-then-softmax","candidate":"paired-row-fused-qk-requant-softmax","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
