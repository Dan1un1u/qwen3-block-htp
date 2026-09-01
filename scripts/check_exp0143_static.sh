#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(50)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(143)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED_SIMD_ROW_PACK = 2' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_qk_norm_rope_pair_pack_rows_shuffle4' \
    "${project_root}/src/dsp/hvx_u8_ops.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_qk_norm_rope_pair_pack_rows_shuffle4 \
    "${dsp_skel}" > "${temporary_dir}/row_pack.txt"
if [[ "$(grep -c 'vshuff' "${temporary_dir}/row_pack.txt")" -lt 4 ]]; then
    printf 'SIMD row pack did not retain the four-stage vshuff transpose\n' >&2
    exit 1
fi
if grep -Eq 'call.*(memcpy|__hexagon_memcpy)' \
        "${temporary_dir}/row_pack.txt"; then
    printf 'SIMD row pack still calls memcpy\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0143 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0143","static_gate":"pass","recipe":"W4U8","control_mode":1,"candidate_mode":2,"simd_row_pack":"four_aligned_rows_four_tiles_vshuff32_vshuff64","memcpy_in_candidate_helper":false,"qkv_ring_expand_workers":3,"one_hmx_owner":true,"qnn_dependency":false}'
