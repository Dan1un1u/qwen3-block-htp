#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(79)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(180)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_QUARTET_TAIL_V5 = 10' include/block_protocol.h
grep -q 'u8_cache_v_quartet_full_tile_rmw_count' include/block_protocol.h
grep -q 'qbh_attention_u8_publish_v_row_group_hvx' include/attention_u8_core.h
grep -q 'qbh_attention_u8_publish_v_row_group_hvx' src/dsp/attention_u8_core.c
grep -q 'qbh_hmx_native_u8_quartet_v_cache_formats' src/dsp/block_imp.c
grep -q 'qbh_scan_cache_dma_2d_write' src/dsp/block_imp.c
grep -q 'quartet_v && fused_short == 0U' src/dsp/block_imp.c
grep -q 'hmx_native_u8_segmented_quartet_v5' src/host/block_main.c
grep -q 'qbh_prepare_quartet_v_reference' src/host/block_main.c
grep -q 'QBH_EXP0180_KV_CACHE_LAYOUT' scripts/run_exp0173.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0180.sh scripts/deploy_exp0180.sh \
    scripts/run_exp0173.sh scripts/run_exp0180.sh \
    scripts/run_exp0180_short_gate.sh scripts/run_exp0180_formal.sh \
    scripts/check_exp0180_static.sh
python3 -m py_compile scripts/summarize_exp0180.py
printf '%s\n' '{"experiment":"EXP-0180","static_gate":"pass","variant":"W4U8","control":"segmented_v4_row_major_mutable_v_tail","candidate":"quartet_native_v5_mutable_v_tail","full_tile_rmw":false,"qnn_dependency":false}'
