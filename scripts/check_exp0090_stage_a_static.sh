#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${project_root}/include/block_protocol.h"
probe="${project_root}/include/probe_protocol.h"
pipeline="${project_root}/src/dsp/w4_parallel_pipeline.c"
host="${project_root}/src/host/block_main.c"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' "${protocol}"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(90)' "${protocol}"
grep -q 'QBH_PROBE_ABI_VERSION UINT32_C(21)' "${probe}"
grep -q 'QBH_BLOCK_W4U8_GATE_UP_QUEUE_AUDIT' "${protocol}"
grep -q 'mlp_activation_queue_wait_ticks' "${pipeline}"
grep -q 'queue->tasks\[slot\]\.compressed_slot = depth' "${pipeline}"
grep -q 'w4u8_gate_up_queue_mode' "${host}"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0090","stage":"A","static_gate":"pass","qnn_dependency":false}\n'
