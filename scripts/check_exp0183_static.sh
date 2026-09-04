#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(82)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(183)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_PARTIAL_VTCM_TAIL_V8 = 13' include/block_protocol.h
grep -q 'QBH_KV_CACHE_HMX_U8_K_VTCM_TAIL_ATLAS_BYTES' include/block_protocol.h
grep -q 'u8_cache_k_vtcm_tail_native_load_bytes' include/block_protocol.h
grep -q 'qbh_hmx_native_u8_partial_k_vtcm_tail_cache_formats' src/dsp/block_imp.c
grep -q 'persistent_k_tail_atlas' src/dsp/block_imp.c
grep -q 'hmx_native_u8_segmented_vtcm_k7_v8' src/host/block_main.c
grep -q 'QBH_EXP0183_KV_CACHE_LAYOUT' scripts/run_exp0173.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0183.sh scripts/deploy_exp0183.sh \
    scripts/run_exp0173.sh scripts/run_exp0183.sh \
    scripts/run_exp0183_short_gate.sh scripts/run_exp0183_formal.sh \
    scripts/check_exp0183_static.sh
python3 -m py_compile scripts/summarize_exp0181.py \
    scripts/summarize_exp0182.py scripts/summarize_exp0183.py
printf '%s\n' '{"experiment":"EXP-0183","static_gate":"pass","variant":"W4U8","control":"EXP-0182-v7-vtcm-v-tail","candidate":"seven-of-eight-vtcm-native-k-tail","external_ddr_journal":"segmented-v4","qnn_dependency":false}'
