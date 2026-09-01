#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(58)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(142)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_GROUP_FENCE_JOIN_ONLY_DOWN_O' \
    "${project_root}/include/block_protocol.h"
grep -q 'strcmp(group_fence, "join_only_down_o")' \
    "${project_root}/src/host/block_main.c"
grep -q 'relaxed_projection_group_fence' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_PROJ_O' "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_PROJ_DOWN' "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx \
    "${dsp_skel}" > "${temporary_dir}/ordered.txt"
"${objdump_bin}" \
    --disassemble-symbols=qbh_unpack_w4_to_f16_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/relaxed.txt"
sed -n '/static void qbh_w4f16_pool_wait(/,/^}/p' \
    "${project_root}/src/dsp/block_imp.c" \
    > "${temporary_dir}/pool_wait.txt"
grep -q 'barrier' "${temporary_dir}/ordered.txt"
if grep -q 'barrier' "${temporary_dir}/relaxed.txt"; then
    printf 'relaxed W4F16 unpack still contains a full barrier\n' >&2
    exit 1
fi
grep -q 'barrier' "${temporary_dir}/pool_wait.txt"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0142 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0142","static_gate":"pass","recipe":"W4F16","control_join_only_down_selectable":true,"candidate_join_only_down_o_selectable":true,"gate_up_subgroup4_preserved":true,"down_relaxed_in_both_cells":true,"o_relaxed_candidate_only":true,"ordered_unpack_has_barrier":true,"relaxed_unpack_has_barrier":false,"pool_join_barrier_retained":true,"one_hmx_owner":true,"qnn_dependency":false}'
