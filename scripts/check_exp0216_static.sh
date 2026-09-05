#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
git diff --check
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(106)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(216)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DIRECT_N_PREFILL_QKVO = 1U << 5' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DIRECT_N_ALL = (1U << 6) - 1U' include/block_protocol.h
grep -q 'logical_rows == QBH_BLOCK_M && past_tokens == 0U' src/dsp/block_imp.c
grep -q 'control) direct_mask=31' scripts/run_exp0216.sh
grep -q 'direct_qkvo) direct_mask=63' scripts/run_exp0216.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0216.sh scripts/deploy_exp0216.sh \
    scripts/run_exp0216.sh scripts/run_exp0216_short_gate.sh \
    scripts/run_exp0216_formal.sh scripts/check_exp0216_static.sh
python3 -m py_compile scripts/summarize_exp0189.py \
    scripts/summarize_exp0215.py scripts/summarize_exp0216.py
printf '%s\n' '{"experiment":"EXP-0216","static_gate":"pass","candidate":"M64-direct-W4-QKVO-plus-MLP","qnn_dependency":false}'
