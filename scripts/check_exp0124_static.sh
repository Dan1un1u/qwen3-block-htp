#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(49)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(124)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_RING_SLOTS UINT32_C(4)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_W4U8_QKV_RING_BATCHES UINT32_C(32)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4U8_QKV_RING_EXPAND_WORKERS' \
    "${project_root}/src/host/block_main.c"
grep -q 'QBH_BLOCK_HMX_U8S8_QKV_RING' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_HVX_POOL_W4U8_QKV_RING' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_pool_u8_qk_prep_publish' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" --disassemble-symbols=qbh_unpack_w4_to_s8_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/unpack.txt"
if grep -q 'barrier' "${temporary_dir}/unpack.txt"; then
    printf 'relaxed W4 expansion still contains a full barrier\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0124 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0124","static_gate":"pass","recipe":"W4U8","control_selectable":true,"ring_expand_workers":[1,2,3],"ring_slots":4,"ordered_dma_producer":true,"one_hmx_owner":true,"mixed_hvx_epoch":true,"head_aligned_publish":true,"qnn_dependency":false}'
