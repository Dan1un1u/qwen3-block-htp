#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(75)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(175)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DECODE_GATE_UP_PAIR_READY2_BATCH16' include/block_protocol.h
grep -q 'output_ready_semaphore' include/w4_parallel_pipeline.h
grep -q 'pair_ready_worker_index' src/dsp/w4_parallel_pipeline.c
grep -q 'pair_ready_worker_count' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_GATE_UP' src/host/block_main.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0175.sh scripts/deploy_exp0175.sh \
    scripts/run_exp0175.sh scripts/check_exp0175_static.sh
printf '%s\n' '{"experiment":"EXP-0175","static_gate":"pass","variant":"W4U8","control":"post_batch8","candidates":["pair_ready_batch8","pair_ready_batch16","pair_ready2_batch8","pair_ready2_batch16"],"qnn_dependency":false}'
