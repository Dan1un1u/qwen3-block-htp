#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(75)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(176)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_O_MAX_BATCH_N_TILES UINT32_C(8)' src/dsp/block_imp.c
grep -q 'w4u8_decode_o_batch_n_tiles' include/block_protocol.h
grep -q 'w4u8_o_batch_n_tiles_observed' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_O_BATCH_N_TILES' src/host/block_main.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0176.sh scripts/deploy_exp0176.sh \
    scripts/run_exp0173.sh scripts/run_exp0176.sh \
    scripts/run_exp0176_short_gate.sh scripts/run_exp0176_formal.sh \
    scripts/check_exp0176_static.sh
python3 -m py_compile scripts/summarize_exp0176_short.py \
    scripts/summarize_exp0176.py
printf '%s\n' '{"experiment":"EXP-0176","static_gate":"pass","variant":"W4U8","control_o_batch":4,"candidate_o_batch":8,"qnn_dependency":false}'
