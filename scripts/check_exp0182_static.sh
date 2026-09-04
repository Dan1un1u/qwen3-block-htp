#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(81)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(182)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_VTCM_TAIL_V7 = 12' include/block_protocol.h
grep -q 'QBH_KV_CACHE_HMX_U8_V_VTCM_TAIL_ATLAS_BYTES' include/block_protocol.h
grep -q 'u8_cache_v_vtcm_tail_native_load_bytes' include/block_protocol.h
grep -q 'qbh_hmx_native_u8_vtcm_tail_v_cache_formats' src/dsp/block_imp.c
grep -q 'persistent_v_tail_atlas' src/dsp/block_imp.c
grep -q 'qbh_hvx_update_v_tail_row' src/dsp/block_imp.c
grep -q 'hmx_native_u8_segmented_vtcm_tail_v7' src/host/block_main.c
grep -q 'QBH_EXP0182_KV_CACHE_LAYOUT' scripts/run_exp0173.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0182.sh scripts/deploy_exp0182.sh \
    scripts/run_exp0173.sh scripts/run_exp0182.sh \
    scripts/run_exp0182_short_gate.sh scripts/run_exp0182_formal.sh \
    scripts/check_exp0182_static.sh
python3 -m py_compile scripts/summarize_exp0181.py \
    scripts/summarize_exp0182.py
printf '%s\n' '{"experiment":"EXP-0182","static_gate":"pass","variant":"W4U8","control":"segmented_v4_row_major_mutable_v_tail","candidate":"prepared_session_vtcm_resident_v7_tail","external_ddr_journal":"segmented_v4","qnn_dependency":false}'
