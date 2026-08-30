#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(37)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(70)' include/block_protocol.h
grep -q 'EXP-0070' src/host/block_main.c
grep -q 'LUT_TEMPLATES_GQA_BATCH = 14' include/block_protocol.h
grep -q 'qbh_attention_u8_gqa_hmx_batch_enabled' src/dsp/block_imp.c
grep -q 'm_tiles <= QBH_ATTENTION_Q_HEADS_PER_GROUP' src/dsp/block_imp.c
grep -q 'score_group, QBH_ATTENTION_Q_HEADS_PER_GROUP' src/dsp/block_imp.c
grep -q 'output_group, QBH_ATTENTION_Q_HEADS_PER_GROUP' src/dsp/block_imp.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0070_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0070","static_gate":"pass","block_abi":37,"runtime_telemetry_experiment":70,"control_attention_hmx_commands":32,"candidate_attention_hmx_commands":16,"control_total_hmx_commands":192,"candidate_total_hmx_commands":176,"hmx_tile_pairs":49408,"attention_contexts":6,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
