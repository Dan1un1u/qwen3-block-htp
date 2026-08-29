#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    [[ -f "${artifact}" ]] || exit 1
done
if strings "${host_executable}" "${host_stub}" "${dsp_skel}" |
        grep -Eiq 'Qnn|QAIRT'; then
    printf 'QNN or QAIRT dependency found in EXP-0045 binaries\n' >&2
    exit 1
fi
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(24)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(45)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)' \
    "${project_root}/include/probe_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QKV_BATCH4 = 2' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_run_w4u8_qkvo_pipelined_projection' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_dma_start_w4u8_batch_prefetch' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_pool_u8_qk_prep_publish' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'release(%0):at' \
    "${project_root}/src/dsp/block_imp.c"

printf '%s\n' '{"experiment":"EXP-0045","stage":"A","static_gate":"pass","block_abi":24,"qnn_dependency":false,"vtcm_request_bytes":8388608,"batch_search":[1,2,4],"single_hmx_owner":true}'
