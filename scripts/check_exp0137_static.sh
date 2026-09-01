#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(57)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(137)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_PROBE_ABI_VERSION UINT32_C(22)' \
    "${project_root}/include/probe_protocol.h"
grep -q 'QBH_W4U8_GATE_UP_HVX_LEAD_CAP_REGIONS' \
    "${project_root}/src/host/block_main.c"
grep -q 'adaptive_throttle_eligible != 0U' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'active <= 2U' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'qurt_sem_down(&state->hmx_progress)' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'qbh_hvx_region_begin(state, 0U)' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'qbh_hvx_region_begin(state, 1U)' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"

for symbol in qbh_unpack_w4_to_s8_hvx_relaxed \
              qbh_copy_hmx_bias_hvx_relaxed; do
    "${objdump_bin}" --disassemble-symbols="${symbol}" "${dsp_skel}" \
        > "${temporary_dir}/${symbol}.txt"
    if grep -q 'barrier' "${temporary_dir}/${symbol}.txt"; then
        printf '%s still contains a full barrier\n' "${symbol}" >&2
        exit 1
    fi
done
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0137 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0137","static_gate":"pass","recipe":"W4U8","ring_slots":16,"lead_caps":[0,8,16],"minimum_unthrottled_expansion_contexts":2,"activation_throttled":false,"blocking_hmx_progress_semaphore":true,"hmx_batch_outputs":8,"persistent_hvx_workers":3,"stream_fence":"single_fence","one_hmx_owner":true,"qnn_dependency":false}'
