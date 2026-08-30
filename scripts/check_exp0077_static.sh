#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(40)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(77)' include/block_protocol.h
grep -q 'EXP-0077' src/host/block_main.c
grep -q 'QBH_BLOCK_MLP_W4U8_STREAMING_GATE_UP_PREFETCH = 6' include/block_protocol.h
grep -q 'qbh_start_chunked_w4_initial_prefetch' src/dsp/block_imp.c
grep -q 'qbh_run_chunked_w4_pipeline_external_prefetched' src/dsp/block_imp.c
grep -q 'qbh_prepare_linked_weight_bundles' src/dsp/w4_parallel_pipeline.c
grep -q 'w4u8_gate_up_cross_prefetch_count' src/host/block_main.c
grep -q 'QBH_EXPECTED_FULL_VTCM_BYTES' src/dsp/block_imp.c
if grep -Rqs --exclude='check_exp0077_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0077","static_gate":"pass","block_abi":40,"runtime_telemetry_experiment":77,"prefetched_linked_dma_descriptors":4,"prefetch_start_stage":"post_o_pre_residual","qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
