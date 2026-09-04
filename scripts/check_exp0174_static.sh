#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(75)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(174)' include/block_protocol.h
grep -q '#define QBH_W4_HMX_MAX_BATCH_OUTPUTS UINT32_C(16)' include/w4_parallel_pipeline.h
grep -q 'qbh_streaming_batch_output_complete' src/dsp/w4_parallel_pipeline.c
grep -q 'QBH_W4U8_DECODE_GATE_UP' src/host/block_main.c
grep -q 'w4u8_mlp_gate_up_inline_slot_release_count' include/block_protocol.h
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0174.sh scripts/deploy_exp0174.sh \
    scripts/run_exp0174.sh scripts/check_exp0174_static.sh \
    scripts/run_exp0173.sh scripts/run_exp0173_formal.sh
printf '%s\n' '{"experiment":"EXP-0174","static_gate":"pass","variant":"W4U8","control":"post_batch8","candidates":["inline_batch8","inline_batch16"],"qnn_dependency":false}'
