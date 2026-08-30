#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${project_root}/include/block_protocol.h"
dsp="${project_root}/src/dsp/block_imp.c"
fp16_ops="${project_root}/src/dsp/hvx_fp16_ops.c"
host="${project_root}/src/host/block_main.c"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(42)' "${protocol}"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(84)' "${protocol}"
grep -q 'QBH_BLOCK_FP16_COMMON_SCHEDULE_INPUT_NORM_POOL' "${protocol}"
grep -q 'QBH_BLOCK_FP16_COMMON_SCHEDULE_POST_RESIDUAL_NORM_POOL' "${protocol}"
grep -q 'qbh_hvx_pool_fp16_input_norm' "${dsp}"
grep -q 'qbh_hvx_pool_fp16_post_residual_norm' "${dsp}"
grep -q 'qbh_hvx_rms_norm_f16_crouton_rows' "${fp16_ops}"
grep -q 'qbh_hvx_residual_rms_norm_f16_crouton_rows' "${fp16_ops}"
grep -q 'input_norm_pool_post_norm_pool' "${host}"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0084","static_gate":"pass","qnn_dependency":false}\n'
