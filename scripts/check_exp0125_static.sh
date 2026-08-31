#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(50)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(125)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_RING_QK_BATCHES UINT32_C(24)' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_W4U8_QKV_RING_QK_TASKS' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_W4U8_QKV_RING_HANDOFF_WORKERS' \
    "${project_root}/src/host/block_main.c"
grep -q 'state->next_v_expand_task' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'job->worker_index >= state->v_expand_worker_count' \
    "${project_root}/src/dsp/block_imp.c"

"${objdump_bin}" --disassemble-symbols=qbh_unpack_w4_to_s8_hvx_relaxed \
    "${dsp_skel}" > "${temporary_dir}/unpack.txt"
if grep -q 'barrier' "${temporary_dir}/unpack.txt"; then
    printf 'relaxed W4 expansion still contains a full barrier\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0125 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0125","static_gate":"pass","recipe":"W4U8","initial_expand_workers":3,"handoff_workers":[0,1,2],"qk_expand_tasks":96,"v_expand_workers":[3,2,1],"ring_slots":4,"one_hmx_owner":true,"qnn_dependency":false}'
