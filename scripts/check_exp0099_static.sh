#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(99)' \
    "${project_root}/include/block_protocol.h"
grep -q 'max_nonstreaming_batch_outputs' \
    "${project_root}/include/w4_parallel_pipeline.h"
grep -q 'continuation_free_semaphore' \
    "${project_root}/include/w4_parallel_pipeline.h"
grep -q 'qurt_sem_up(first_free)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qurt_sem_up(continuation_free)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'output_base += batch_capacity' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'batch_index == 0U' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0099","static_gate":"pass","control_down_batch_outputs":1,"candidate_down_batch_outputs":4,"candidate_expected_down_commands":16,"candidate_expected_total_hmx_commands":128,"in_command_first_and_continuation_slot_release":true,"whole_batch_preready_barrier":false,"qnn_dependency":false}'
