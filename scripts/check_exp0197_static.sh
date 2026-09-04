#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(92)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(197)' include/block_protocol.h
grep -q 'w4u8_decode_lm_head_group_tiles != 64U' src/dsp/block_imp.c
grep -q 'w4u8_decode_lm_head_group_tiles != 64U' src/host/block_main.c
grep -q '8|16|32|64' scripts/run_exp0173.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_MASK=15' scripts/run_exp0197.sh
grep -q 'group_tiles=64' scripts/run_exp0197.sh
grep -q 'compressed_slots\[0\] = buffers->expanded_weight' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0197.sh scripts/deploy_exp0197.sh \
    scripts/run_exp0197.sh scripts/run_exp0197_short_gate.sh \
    scripts/run_exp0197_formal.sh scripts/check_exp0197_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0196.py
printf '%s\n' '{"experiment":"EXP-0197","static_gate":"pass","candidate":"direct-n-decode-lm-head-batch64","qnn_dependency":false}'
