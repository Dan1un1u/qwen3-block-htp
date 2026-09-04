#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(95)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(200)' include/block_protocol.h
grep -q 'qbh_start_w4u8_direct_n_gate_prefetch' src/dsp/block_imp.c
grep -q 'qbh_consume_w4u8_direct_n_gate_prefetch' src/dsp/block_imp.c
grep -q 'state->weight_slot = buffers->expanded_weight' src/dsp/block_imp.c
grep -q 'state->bias_slot = buffers->rope_cos' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0200.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES=16' scripts/run_exp0200.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0200.sh scripts/deploy_exp0200.sh \
    scripts/run_exp0200.sh scripts/run_exp0200_short_gate.sh \
    scripts/run_exp0200_formal.sh scripts/check_exp0200_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0200.py
printf '%s\n' '{"experiment":"EXP-0200","static_gate":"pass","candidate":"O-to-Gate-direct-n-cross-prefetch","qnn_dependency":false}'
