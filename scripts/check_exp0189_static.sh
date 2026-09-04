#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(86)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(189)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_SWIGLU_DECODE_ROWS UINT32_C(4)' include/block_protocol.h
grep -q 'w4u8_decode_swiglu_rows' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'qbh_w4u8_poison_swiglu_padding' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_SWIGLU_ROWS' src/host/block_main.c scripts/run_exp0173.sh
grep -q 'QBH_EXP0189_SWIGLU_ROWS' scripts/run_exp0173.sh scripts/run_exp0189.sh
grep -q 'QBH_BLOCK_W4U8_DECODE_PROJECTION_DIRECT_N = 1' include/block_protocol.h
grep -q 'qbh_hmx_accumulate_u8n4_projection' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0189.sh scripts/deploy_exp0189.sh \
    scripts/run_exp0189.sh scripts/run_exp0189_short_gate.sh \
    scripts/run_exp0189_formal.sh scripts/check_exp0189_static.sh
python3 -m py_compile scripts/summarize_exp0189.py
printf '%s\n' '{"experiment":"EXP-0189","static_gate":"pass","candidate":"direct-n-decode-row4-swiglu","qnn_dependency":false}'
