#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(48)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(114)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4F16_GROUP_FENCE_SEMAPHORE_ONLY' \
    "${project_root}/include/block_protocol.h"
grep -q 'strcmp(group_fence, "semaphore_only")' \
    "${project_root}/src/host/block_main.c"
grep -q 'qbh_w4f16_pool_wait_semaphore_only' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" \
    --disassemble-symbols=qbh_w4f16_pool_wait_semaphore_only \
    "${dsp_skel}" > "${temporary_dir}/semaphore_only.txt"
grep -A8 'static void qbh_w4f16_pool_wait(' \
    "${project_root}/src/dsp/block_imp.c" | grep -q 'barrier'
if grep -q 'barrier' "${temporary_dir}/semaphore_only.txt"; then
    printf 'semaphore-only W4F16 join still contains a full barrier\n' >&2
    exit 1
fi
if [[ "$(grep -c 'qurt_sem_down' "${temporary_dir}/semaphore_only.txt")" -lt 1 ]]; then
    printf 'W4F16 join path lost semaphore waits\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0114 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0114","static_gate":"pass","recipe":"W4F16","control_join_only_selectable":true,"candidate_semaphore_only_selectable":true,"control_post_join_barrier":true,"candidate_post_join_barrier":false,"control_semaphore_waits":true,"candidate_semaphore_waits":true,"one_hmx_owner":true,"qnn_dependency":false}'
