#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(83)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(184)' include/block_protocol.h
grep -q 'qbh_attention_u8_update_k_native_row_hvx' include/attention_u8_core.h src/dsp/attention_u8_core.c src/dsp/block_imp.c
grep -q 'Q6_vscatter_RMVwV' src/dsp/attention_u8_core.c
grep -q 'u8_cache_k_vtcm_tail_hvx_row_update_count' include/block_protocol.h src/host/block_main.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2; exit 1
fi
bash -n scripts/build_exp0184.sh scripts/deploy_exp0184.sh scripts/run_exp0184.sh \
    scripts/run_exp0184_short_gate.sh scripts/run_exp0184_formal.sh scripts/check_exp0184_static.sh
python3 -m py_compile scripts/summarize_exp0184.py
printf '%s\n' '{"experiment":"EXP-0184","static_gate":"pass","candidate":"EXP-0183-K13-plus-HVX-row-update","qnn_dependency":false}'
