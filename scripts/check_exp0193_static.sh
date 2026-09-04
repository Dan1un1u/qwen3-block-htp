#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(89)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(193)' include/block_protocol.h
grep -q 'w4u8_decode_direct_n_down_dma_batch_n_tiles' \
    include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_DMA_BATCH_N_TILES' \
    scripts/run_exp0173.sh scripts/run_exp0193.sh
grep -q 'dma_batch_tiles, uint32_t hmx_batch_tiles' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0193.sh scripts/deploy_exp0193.sh \
    scripts/run_exp0193.sh scripts/run_exp0193_short_gate.sh \
    scripts/run_exp0193_formal.sh scripts/check_exp0193_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0190.py
printf '%s\n' '{"experiment":"EXP-0193","static_gate":"pass","candidate":"direct-n-decode-down-dma4-hmx2","qnn_dependency":false}'
