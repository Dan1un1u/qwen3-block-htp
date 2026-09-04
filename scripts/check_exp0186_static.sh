#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(85)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(186)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_SESSION_VTCM_TAIL_V10 = 15' include/block_protocol.h
grep -q 'qbh_hmx_native_u8_session_v_vtcm_tail_cache_formats' src/dsp/block_imp.c
grep -q 'qbh_hvx_update_v_tail_native_row' src/dsp/block_imp.c
grep -q 'u8_cache_v_vtcm_tail_direct_row_update_count' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'u8_cache_v_vtcm_tail_direct_row_update_ticks' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'u8_cache_v_vtcm_tail_ddr_write_skip_count' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'u8_cache_v_vtcm_tail_ddr_write_skip_bytes' include/block_protocol.h src/dsp/block_imp.c src/host/block_main.c
grep -q 'hmx_native_u8_segmented_vtcm_kv_session_v10' src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0186.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2; exit 1
fi
bash -n scripts/build_exp0186.sh scripts/deploy_exp0186.sh scripts/run_exp0186.sh \
    scripts/run_exp0186_short_gate.sh scripts/run_exp0186_formal.sh scripts/check_exp0186_static.sh
python3 -m py_compile scripts/summarize_exp0186.py
printf '%s\n' '{"experiment":"EXP-0186","static_gate":"pass","candidate":"session-native-K14-V15-direct-V-append-no-DDR-journal","qnn_dependency":false}'
