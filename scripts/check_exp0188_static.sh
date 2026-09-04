#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(85)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(188)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DECODE_PROJECTION_DIRECT_N = 1' include/block_protocol.h
grep -q 'qbh_hmx_accumulate_u8n4_projection' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_PROJECTION_MODE' src/host/block_main.c scripts/run_exp0173.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_MASK' src/host/block_main.c scripts/run_exp0173.sh
grep -q 'direct_n_weight_offset' include/block_protocol.h src/host/block_main.c src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2; exit 1
fi
bash -n scripts/build_exp0188.sh scripts/deploy_exp0188.sh \
    scripts/run_exp0188.sh scripts/run_exp0188_formal.sh \
    scripts/check_exp0188_static.sh
python3 -m py_compile scripts/summarize_exp0188.py
printf '%s\n' '{"experiment":"EXP-0188","static_gate":"pass","candidate":"decode-only-direct-n-full-stack","qnn_dependency":false}'
