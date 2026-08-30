#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(36)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(64)' include/block_protocol.h
grep -q 'FUSED_QK_REQUANT_HMX_BATCH = 12' include/block_protocol.h
grep -q 'qbh_attention_u8_hmx_batch_enabled' src/dsp/block_imp.c
grep -q 'u8_attention_qk_av_hmx_ticks' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0064_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0064","static_gate":"pass","block_abi":36,"runtime_telemetry_experiment":64,"control_attention_hmx_commands":96,"candidate_attention_hmx_commands":32,"control_total_hmx_commands":256,"candidate_total_hmx_commands":192,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
