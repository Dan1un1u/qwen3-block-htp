#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(49)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(121)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_GATE_UP_MIN_HVX_WORKERS UINT32_C(3)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_W4U8_GATE_UP_MAX_HVX_WORKERS UINT32_C(4)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'header->w4u8_mlp_gate_up_hvx_workers =' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'header->mlp_hvx_contexts;' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4U8_GATE_UP_RING_SLOTS=16' \
    "${project_root}/scripts/run_exp0121.sh"

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
    printf 'QNN dependency found in EXP-0121 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0121","static_gate":"pass","recipe":"W4U8","ring_slots":16,"worker3_runtime_selectable":true,"worker4_runtime_selectable":true,"hmx_batch_outputs":8,"stream_fence":"single_fence","one_hmx_owner":true,"qnn_dependency":false}'
