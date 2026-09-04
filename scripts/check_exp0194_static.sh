#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(89)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(194)' include/block_protocol.h
grep -q 'w4u8_decode_direct_n_qkv_batch_n_tiles' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0194.sh
grep -q 'state.tiles_per_batch == 8U' src/dsp/block_imp.c
grep -q 'buffers->input_norm_weight' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0194.sh scripts/deploy_exp0194.sh \
    scripts/run_exp0194.sh scripts/run_exp0194_short_gate.sh \
    scripts/run_exp0194_formal.sh scripts/check_exp0194_static.sh
python3 -m py_compile scripts/summarize_exp0189.py \
    scripts/summarize_exp0194.py
printf '%s\n' '{"experiment":"EXP-0194","static_gate":"pass","candidate":"direct-n-decode-qkv-head-pair-batch8","qnn_dependency":false}'
