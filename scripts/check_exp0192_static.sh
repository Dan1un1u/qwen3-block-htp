#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(89)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(192)' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES' scripts/run_exp0173.sh scripts/run_exp0192.sh
grep -q 'header->w4u8_down_hmx_batch_outputs) != 0' src/dsp/block_imp.c
grep -q 'header->w4u8_down_hmx_batch_outputs;' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0192.sh scripts/deploy_exp0192.sh \
    scripts/run_exp0192.sh scripts/run_exp0192_short_gate.sh \
    scripts/run_exp0192_formal.sh scripts/check_exp0192_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0190.py
printf '%s\n' '{"experiment":"EXP-0192","static_gate":"pass","candidate":"direct-n-decode-down-batch4","qnn_dependency":false}'
