#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
git diff --check
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(105)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(215)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DIRECT_N_PREFILL_MLP = 1U << 4' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DIRECT_N_DECODE_ALL = (1U << 4) - 1U' include/block_protocol.h
grep -q 'header->logical_m == QBH_BLOCK_M' src/dsp/block_imp.c
grep -q 'prefill_direct_projections' scripts/summarize_exp0189.py
grep -q 'control) direct_mask=15' scripts/run_exp0215.sh
grep -q 'direct_mlp) direct_mask=31' scripts/run_exp0215.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0215.sh scripts/deploy_exp0215.sh \
    scripts/run_exp0215.sh scripts/run_exp0215_short_gate.sh \
    scripts/run_exp0215_formal.sh scripts/check_exp0215_static.sh
python3 -m py_compile scripts/summarize_exp0189.py \
    scripts/summarize_exp0215.py
printf '%s\n' '{"experiment":"EXP-0215","static_gate":"pass","candidate":"M64-direct-W4-Gate-Up-Down","qnn_dependency":false}'
