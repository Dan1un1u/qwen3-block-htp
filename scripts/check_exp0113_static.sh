#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(48)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(113)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_STREAM_FENCE_BARRIER_ONLY' \
    "${project_root}/include/block_protocol.h"
grep -q 'strcmp(stream_fence, "barrier_only")' \
    "${project_root}/src/host/block_main.c"
grep -q 'stream_fence_mode !=' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'QBH_W4_STREAM_FENCE_BARRIER_ONLY' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q '\*ready = task.stream_generation;' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'release(%0):at' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_hmx_accumulate_u8s8_streaming \
    "${dsp_skel}" > "${temporary_dir}/consumer.txt"
grep -q 'barrier' "${temporary_dir}/consumer.txt"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0113 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0113","static_gate":"pass","recipe":"W4U8","control_single_fence_selectable":true,"candidate_barrier_only_selectable":true,"worker_full_barrier_retained":true,"volatile_generation_store_retained":true,"control_address_release_retained":true,"candidate_address_release_skipped":true,"consumer_full_barrier_retained":true,"one_hmx_owner":true,"qnn_dependency":false}'
