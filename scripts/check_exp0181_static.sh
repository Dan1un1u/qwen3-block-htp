#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(80)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(181)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_ATTENTION_PUBLISH_V6 = 11' include/block_protocol.h
grep -q 'u8_cache_v_quartet_attention_publish_count' include/block_protocol.h
grep -q 'qbh_hmx_native_u8_attention_publish_v_cache_formats' src/dsp/block_imp.c
grep -q 'The Attention consumer already has to load that group' src/dsp/block_imp.c
grep -q 'The last complete group is still raw on V6' src/dsp/block_imp.c
grep -q 'qbh_scan_cache_dma_2d_write' src/dsp/block_imp.c
grep -q 'attention_publish_v && partial_rows == 0U' src/dsp/block_imp.c
grep -q 'hmx_native_u8_segmented_attention_publish_v6' src/host/block_main.c
grep -q 'QBH_EXP0181_KV_CACHE_LAYOUT' scripts/run_exp0173.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0181.sh scripts/deploy_exp0181.sh \
    scripts/run_exp0173.sh scripts/run_exp0181.sh \
    scripts/check_exp0181_static.sh
printf '%s\n' '{"experiment":"EXP-0181","static_gate":"pass","variant":"W4U8","control":"segmented_v4_row_major_mutable_v_tail","candidate":"attention_publish_v6_mutable_v_tail","append_side_group_read":false,"full_tile_rmw":false,"qnn_dependency":false}'
