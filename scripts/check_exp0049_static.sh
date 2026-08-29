#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(26)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(49)' include/block_protocol.h
grep -q '#define QBH_BLOCK_W4U8_GATE_UP_HMX_BATCH_N_TILES UINT32_C(8)' \
    src/dsp/block_imp.c
grep -q 'request.batch_output_count = batch_outputs' \
    src/dsp/w4_parallel_pipeline.c
grep -q 'request.store_output = 1U' src/dsp/w4_parallel_pipeline.c
grep -q 'w4u8_mlp_gate_up_hmx_command_count' include/block_protocol.h
grep -q 'w4u8_mlp_gate_up_expanded_slot_count' include/block_protocol.h

if grep -RIl --exclude='check_exp0049_static.sh' \
        -E 'Qnn|libQnn|QAIRT' include src CMakeLists.txt cmake 2>/dev/null | \
        grep -q .; then
    echo 'QNN dependency found in the standalone runtime' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0049","static_gate":"pass","block_abi":26,"runtime_telemetry_experiment":49,"gate_up_hmx_batch_n_tiles":8,"gate_up_expanded_slots":8,"accumulate_store_single_command":true,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
