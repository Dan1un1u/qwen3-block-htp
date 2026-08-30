#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(32)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(59)' include/block_protocol.h
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP_VGATHER_VDEAL_PAIRED_SOFTMAX = 11' include/block_protocol.h
grep -q 'qbh_attention_u8_softmax_group_paired_rows' include/attention_u8_core.h
grep -q 'qbh_attention_u8_softmax_group_paired_rows' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_pack_v_native_vgather_vdeal' src/dsp/attention_u8_core.c
test "$(grep -c 'recentered = Q6_Vb_vdeal_Vb(recentered)' src/dsp/attention_u8_core.c)" -eq 5
grep -q 'u8_log2_gqa_qkv_overlap_vgather_vdeal_paired_softmax' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0059_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0059","static_gate":"pass","block_abi":32,"runtime_telemetry_experiment":59,"control":"EXP-0058-vgather-vdeal-independent-softmax","candidate":"EXP-0058-vgather-vdeal-paired-softmax","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
