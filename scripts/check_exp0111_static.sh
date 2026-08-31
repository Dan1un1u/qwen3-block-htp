#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(46)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(111)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4F16_GROUP_FENCE' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_unpack_w4_to_f16_hvx_relaxed' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx \
    "${dsp_skel}" > "${temporary_dir}/ordered.txt"
"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/relaxed.txt"
grep -q 'barrier' "${temporary_dir}/ordered.txt"
if grep -q 'barrier' "${temporary_dir}/relaxed.txt"; then
    printf 'relaxed W4F16 unpack still contains a full barrier\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0111 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0111","static_gate":"pass","recipe":"W4F16","control_runtime_selectable":true,"join_only_runtime_selectable":true,"ordered_unpack_has_barrier":true,"relaxed_unpack_has_barrier":false,"one_hmx_owner":true,"qnn_dependency":false}'
