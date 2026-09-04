#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(90)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(195)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DIRECT_N_MAX_BATCH_N_TILES UINT32_C(32)' src/dsp/block_imp.c
grep -q 'batch_tiles == 32U' src/dsp/block_imp.c
grep -q '? buffers->post_norm_weight' src/dsp/block_imp.c
grep -q 'w4u8_decode_direct_n_gate_up_batch_n_tiles != 32U' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0195.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0195.sh scripts/deploy_exp0195.sh \
    scripts/run_exp0195.sh scripts/run_exp0195_short_gate.sh \
    scripts/run_exp0195_formal.sh scripts/check_exp0195_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0190.py
printf '%s\n' '{"experiment":"EXP-0195","static_gate":"pass","candidate":"direct-n-decode-gate-up-batch32","qnn_dependency":false}'
