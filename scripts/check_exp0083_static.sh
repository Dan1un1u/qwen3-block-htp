#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(42)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(83)' include/block_protocol.h
grep -q 'EXP-0083' src/host/block_main.c
grep -q '#define QBH_W4_HMX_MAX_BATCH_OUTPUTS UINT32_C(32)' include/w4_parallel_pipeline.h
grep -q '#define QBH_W4_MAX_EXPANDED_SLOT_COUNT UINT32_C(32)' include/probe_protocol.h
grep -q 'return pair_count < 8U ? 8U : pair_count;' src/dsp/block_imp.c
grep -q 'qbh_w4u8_gate_up_pair_slot_count(batch_n_tiles) \* 2U' src/dsp/block_imp.c
grep -q 'qbh_w4u8_gate_up_pair_slot_count(' src/dsp/block_imp.c
grep -q 'layout->expanded_slot_count = batch_n_tiles' src/dsp/block_imp.c
grep -q 'w4u8_mlp_gate_up_hmx_command_count' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0083_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0083","static_gate":"pass","block_abi":42,"runtime_telemetry_experiment":83,"gate_up_batch_candidates":[8,16,32],"gate_up_pair_slots":{"8":8,"16":8,"32":16},"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
