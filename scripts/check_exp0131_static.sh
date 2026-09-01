#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(50)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(131)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_TAIL_PREP_MAIN_DRAIN' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4U8_QKV_TAIL_PREP' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_attention_u8_qk_prep_pool_run_tasks' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_HMX_U8S8_QKV_RING' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" --disassemble-symbols=qbh_unpack_w4_to_s8_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/unpack.txt"
if grep -q 'barrier' "${temporary_dir}/unpack.txt"; then
    printf 'relaxed W4 expansion still contains a full barrier\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0131 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0131","static_gate":"pass","recipe":"W4U8","control_selectable":true,"tail_prep_modes":["control","main_drain"],"ring_expand_workers":3,"post_dma_only":true,"one_hmx_owner":true,"qnn_dependency":false}'
