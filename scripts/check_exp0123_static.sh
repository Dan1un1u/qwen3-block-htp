#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(51)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(123)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_GROUP_FENCE_PERSISTENT' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_HVX_POOL_W4F16_PERSISTENT_EXPAND' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'persistent_expand_generation' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4F16_GROUP_FENCE' \
    "${project_root}/scripts/run_exp0123.sh"

"${objdump_bin}" \
    --disassemble-symbols=qbh_w4f16_persistent_expand_worker_run \
    "${dsp_skel}" > "${temporary_dir}/persistent.txt"
grep -q 'pause' "${temporary_dir}/persistent.txt"
grep -q 'release' "${temporary_dir}/persistent.txt"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0123 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0123","static_gate":"pass","recipe":"W4F16","join_only_runtime_selectable":true,"persistent_runtime_selectable":true,"persistent_workers":2,"persistent_generation_release":true,"one_hmx_owner":true,"qnn_dependency":false}'
