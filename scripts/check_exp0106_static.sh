#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(44)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(106)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_RESIDUAL_HVX_FUSED_POST_NORM_POOL6_SHUFFLE4' \
    "${project_root}/include/block_protocol.h"
grep -q 'DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_hvx_residual_add_u8_native_output_rows_shuffle4' \
    "${project_root}/src/dsp/hvx_u8_ops.c"
grep -q 'qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates_shuffle4' \
    "${project_root}/src/dsp/attention_u8_core.c"
grep -q 'qbh_qk_norm_rope_two_heads_u8_quarter_tiled' \
    "${project_root}/src/dsp/hvx_u8_ops.c"
grep -q 'w4u8_down_hmx_batch_outputs' \
    "${project_root}/src/dsp/block_imp.c"
if grep -q 'qbh_hvx_qk_norm_rope_u8_native_dual_head' \
        "${project_root}/src/dsp/hvx_u8_ops.c"; then
    printf 'rejected EXP-0105 dual-head helper found\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0106","static_gate":"pass","runtime_selectable_cells":true,"down_batch4":true,"softmax_shuffle4":true,"residual_shuffle4":true,"qk_quarter_tiled":true,"rejected_exp0105_helper":false,"qnn_dependency":false}'
