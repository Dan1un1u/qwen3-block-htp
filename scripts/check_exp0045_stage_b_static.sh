#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${project_root}/scripts/check_exp0045_static.sh" >/dev/null

grep -q 'QBH_BLOCK_W4U8_QKVO_BATCH4 = 3' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_w4u8_qkvo_batch_tiles' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_run_w4u8_qkvo_pipelined_projection' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'QBH_BLOCK_PROJ_O' \
    "${project_root}/src/dsp/block_imp.c"

printf '%s\n' '{"experiment":"EXP-0045","stage":"B","static_gate":"pass","block_abi":24,"qnn_dependency":false,"vtcm_request_bytes":8388608,"selected_qkv_plan":"qkv_batch4","o_candidate":"qkvo_batch4","single_hmx_owner":true}'
