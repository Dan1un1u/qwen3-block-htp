#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(77)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(178)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_COMMON_OP_DECODE_ROWS UINT32_C(4)' include/block_protocol.h
grep -q 'w4u8_decode_common_op_rows' include/block_protocol.h
grep -q 'w4u8_common_op_rows_observed' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_COMMON_OP_ROWS' src/host/block_main.c
grep -q 'qbh_hvx_rms_norm_u8_native_activation_rows' src/dsp/block_imp.c
grep -q 'qbh_hvx_residual_rms_norm_u8_native_io_rows_shuffle4' src/dsp/block_imp.c
grep -q 'qbh_hvx_residual_add_u8_native_output_rows_shuffle4' src/dsp/block_imp.c
grep -q 'scan_dynamic_attention != 0U && logical_rows == 1U' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0178.sh scripts/deploy_exp0178.sh \
    scripts/run_exp0173.sh scripts/run_exp0178.sh \
    scripts/run_exp0178_short_gate.sh scripts/run_exp0178_formal.sh \
    scripts/check_exp0178_static.sh
python3 -m py_compile scripts/summarize_exp0178_short.py \
    scripts/summarize_exp0178.py
printf '%s\n' '{"experiment":"EXP-0178","static_gate":"pass","variant":"W4U8","control_rows":64,"candidate_rows":4,"target_ops":["Input_RMSNorm","PostAttention_residual_RMSNorm","Final_residual"],"padding_poison_audit":true,"qnn_dependency":false}'
