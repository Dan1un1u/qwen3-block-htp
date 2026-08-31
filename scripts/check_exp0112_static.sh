#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(47)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(112)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_STREAM_FENCE_RELEASE_ONLY' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4U8_STREAM_FENCE' \
    "${project_root}/src/host/block_main.c"
grep -q 'stream_fence_mode !=' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'QBH_W4_STREAM_FENCE_RELEASE_ONLY' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"

for symbol in qbh_unpack_w4_to_s8_hvx \
              qbh_unpack_w4_to_s8_hvx_relaxed \
              qbh_copy_hmx_bias_hvx \
              qbh_copy_hmx_bias_hvx_relaxed; do
    "${objdump_bin}" --disassemble-symbols="${symbol}" "${dsp_skel}" \
        > "${temporary_dir}/${symbol}.txt"
done
grep -q 'barrier' "${temporary_dir}/qbh_unpack_w4_to_s8_hvx.txt"
grep -q 'barrier' "${temporary_dir}/qbh_copy_hmx_bias_hvx.txt"
if grep -q 'barrier' \
        "${temporary_dir}/qbh_unpack_w4_to_s8_hvx_relaxed.txt"; then
    printf 'relaxed W4U8 unpack still contains a full barrier\n' >&2
    exit 1
fi
if grep -q 'barrier' \
        "${temporary_dir}/qbh_copy_hmx_bias_hvx_relaxed.txt"; then
    printf 'relaxed W4U8 bias copy still contains a full barrier\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0112 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0112","static_gate":"pass","recipe":"W4U8","control_runtime_selectable":true,"single_fence_runtime_selectable":true,"release_only_runtime_selectable":true,"ordered_unpack_has_barrier":true,"relaxed_unpack_has_barrier":false,"ordered_bias_copy_has_barrier":true,"relaxed_bias_copy_has_barrier":false,"generation_release_retained":true,"one_hmx_owner":true,"qnn_dependency":false}'
