#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(93)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(198)' include/block_protocol.h
grep -q 'qbh_run_w4u8_direct_n_gate_up_pair' src/dsp/block_imp.c
grep -q 'Flatten their six batch-32' src/dsp/block_imp.c
grep -q 'w4u8_decode_direct_n_gate_up_continuous != 0U' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0198.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_MASK=15' scripts/run_exp0198.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES=32' \
    scripts/run_exp0198.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0198.sh scripts/deploy_exp0198.sh \
    scripts/run_exp0198.sh scripts/run_exp0198_short_gate.sh \
    scripts/run_exp0198_formal.sh scripts/check_exp0198_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0198.py
printf '%s\n' '{"experiment":"EXP-0198","static_gate":"pass","candidate":"continuous-direct-n-gate-to-up-ring","qnn_dependency":false}'
