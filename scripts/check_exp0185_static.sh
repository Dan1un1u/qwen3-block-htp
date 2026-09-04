#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(84)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(185)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_SESSION_VTCM_TAIL_V9 = 14' include/block_protocol.h
grep -q 'qbh_hmx_native_u8_session_k_vtcm_tail_cache_formats' src/dsp/block_imp.c
grep -q 'u8_cache_k_vtcm_tail_ddr_write_skip_count' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'u8_cache_k_vtcm_tail_ddr_write_skip_bytes' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'hmx_native_u8_segmented_vtcm_k7_session_v9' src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0185.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2; exit 1
fi
bash -n scripts/build_exp0185.sh scripts/deploy_exp0185.sh scripts/run_exp0185.sh \
    scripts/run_exp0185_short_gate.sh scripts/run_exp0185_formal.sh scripts/check_exp0185_static.sh
python3 -m py_compile scripts/summarize_exp0183.py scripts/summarize_exp0185.py
printf '%s\n' '{"experiment":"EXP-0185","static_gate":"pass","candidate":"session-native-K14-no-cached-head-DDR-journal","qnn_dependency":false}'
