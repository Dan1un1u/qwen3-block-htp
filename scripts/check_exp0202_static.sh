#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(97)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(202)' include/block_protocol.h
grep -q 'qbh_prefetch_w4u8_direct_n_down_first_group' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_PREFETCH' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0202.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES=16' scripts/run_exp0202.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0202.sh scripts/deploy_exp0202.sh \
    scripts/run_exp0202.sh scripts/run_exp0202_short_gate.sh \
    scripts/run_exp0202_formal.sh scripts/check_exp0202_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0202.py
printf '%s\n' '{"experiment":"EXP-0202","static_gate":"pass","candidate":"first-Down-batch2-cross-prefetch","qnn_dependency":false}'
