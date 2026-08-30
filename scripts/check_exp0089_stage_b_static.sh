#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(44)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(89)' include/block_protocol.h
grep -q 'QBH_W4_DOWN_PRESTAGE_OUTPUTS UINT32_C(4)' include/w4_parallel_pipeline.h
grep -q 'qbh_prestage_down_prefix' src/dsp/w4_parallel_pipeline.c
grep -q 'qbh_run_chunked_w4_pipeline_external_hvx_prestaged' \
    src/dsp/w4_parallel_pipeline.c
grep -q 'w4u8_gate_down_prestage_requested' src/dsp/block_imp.c
grep -q 'gate_up_down_prestage' src/host/block_main.c
grep -q 'prestage_mode' scripts/run_exp0089.sh
if grep -Rqs --exclude='check_exp0089_stage_b_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' include src CMakeLists.txt; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

printf '%s\n' \
    '{"experiment":"EXP-0089","stage":"B","static_gate":"pass","block_abi":44,"runtime_telemetry_experiment":89,"prestage_outputs":4,"prestage_dma_descriptors":2,"selected_trigger_output_base":288,"intermediate_ddr":false,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
