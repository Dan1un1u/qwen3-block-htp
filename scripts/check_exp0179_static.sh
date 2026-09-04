#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(78)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(179)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QK_PREP_DECODE_ROWS UINT32_C(4)' include/block_protocol.h
grep -q 'w4u8_decode_qk_norm_rope_rows' include/block_protocol.h
grep -q 'w4u8_decode_qk_padding_poison' include/block_protocol.h
grep -q 'w4u8_decode_k_temp_carrier_skipped_count' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_QK_NORM_ROPE_ROWS' src/host/block_main.c
grep -q 'QBH_W4U8_DECODE_QK_PADDING_POISON' src/host/block_main.c
grep -q 'qbh_hvx_qk_norm_rope_u8_native_head_pair_rows' src/dsp/block_imp.c
grep -q 'qbh_hvx_poison_u8_native_head_pair_padding' src/dsp/block_imp.c
grep -q 'rows % 4U != 0U' src/dsp/hvx_u8_ops.c
grep -q 'job->u8_decode_k_temp_carrier_skipped_count += 2U' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0179.sh scripts/deploy_exp0179.sh \
    scripts/run_exp0173.sh scripts/run_exp0179.sh \
    scripts/run_exp0179_short_gate.sh scripts/run_exp0179_formal.sh \
    scripts/check_exp0179_static.sh
python3 -m py_compile scripts/summarize_exp0179.py
printf '%s\n' '{"experiment":"EXP-0179","static_gate":"pass","variant":"W4U8","control_qk_rows":64,"candidate_qk_rows":4,"decode_only":true,"skip_unused_k_temp_carrier":true,"padding_poison_audit":true,"qnn_dependency":false}'
