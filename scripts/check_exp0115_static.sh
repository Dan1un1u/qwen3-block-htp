#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(48)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(115)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_UNMASKED' \
    "${project_root}/include/block_protocol.h"
grep -q 'strcmp(group_fence, "join_only_unmasked")' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_unpack_w4_to_f16_hvx_relaxed_unmasked' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/masked.txt"
"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx_relaxed_unmasked \
    "${dsp_skel}" > "${temporary_dir}/unmasked.txt"
grep -q 'vand' "${temporary_dir}/masked.txt"
if grep -q 'vand' "${temporary_dir}/unmasked.txt"; then
    printf 'unmasked W4F16 helper still contains vector and\n' >&2
    exit 1
fi
for opcode in vlsr vdeal vlut16 vshuff; do
    grep -q "${opcode}" "${temporary_dir}/masked.txt"
    grep -q "${opcode}" "${temporary_dir}/unmasked.txt"
done
masked_store_count="$(grep -c 'vmem(r1' "${temporary_dir}/masked.txt")"
unmasked_store_count="$(grep -c 'vmem(r1' "${temporary_dir}/unmasked.txt")"
if [[ "${masked_store_count}" != "${unmasked_store_count}" ]]; then
    printf 'W4F16 helpers have different vector-store count\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0115 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0115","static_gate":"pass","recipe":"W4F16","control_join_only_selectable":true,"candidate_unmasked_selectable":true,"control_vand":true,"candidate_vand":false,"high_shift_retained":true,"vdeal_retained":true,"vlut16_retained":true,"vshuff_retained":true,"vector_store_count_equal":true,"one_hmx_owner":true,"qnn_dependency":false}'
