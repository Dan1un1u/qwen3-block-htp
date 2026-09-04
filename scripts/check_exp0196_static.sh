#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(91)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(196)' include/block_protocol.h
grep -q '? header->w4u8_decode_lm_head_group_tiles' src/dsp/block_imp.c
grep -q 'compressed_slots\[0\] = buffers->expanded_weight' src/dsp/block_imp.c
grep -q 'compressed_slots\[1\] = buffers->expanded_weight_alt' src/dsp/block_imp.c
grep -q 'group_limit > QBH_BLOCK_W4U8_DIRECT_N_SAFE_BATCH_N_TILES' src/dsp/block_imp.c
grep -q 'w4u8_decode_lm_head_group_tiles != 32U' src/dsp/block_imp.c
grep -q 'w4u8_decode_lm_head_group_tiles != 32U' src/host/block_main.c
grep -q '8|16|32' scripts/run_exp0173.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_MASK="${direct_mask}"' scripts/run_exp0196.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES=32' scripts/run_exp0196.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0196.sh scripts/deploy_exp0196.sh \
    scripts/run_exp0196.sh scripts/run_exp0196_short_gate.sh \
    scripts/run_exp0196_formal.sh scripts/check_exp0196_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0196.py
printf '%s\n' '{"experiment":"EXP-0196","static_gate":"pass","candidate":"direct-n-decode-lm-head-batch32","qnn_dependency":false}'
