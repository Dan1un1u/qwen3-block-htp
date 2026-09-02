#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(69)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(161)' include/block_protocol.h
grep -q 'reference_kv_cache_k_hmx_%s.bin' src/host/block_main.c
grep -q 'reference_kv_cache_v_hmx_%s.bin' src/host/block_main.c
grep -q 'qbh_scan_u8_attention_delta_pipeline' src/dsp/block_imp.c
grep -q 'first < decode_rows' src/dsp/block_imp.c
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_SEGMENTED_V4' include/block_protocol.h
grep -q 'QBH_KV_CACHE_HMX_U8_V_SEGMENT_BLOCK_SEGMENTS UINT32_C(32)' include/block_protocol.h
grep -q 'QBH_BLOCK_U8_SEGMENT_RING_SLOTS UINT32_C(26)' src/dsp/block_imp.c
grep -q 'qbh_scan_u8_attention_segmented_short_pipeline' src/dsp/block_imp.c
grep -q 'qbh_scan_run_u8_segmented_async_phase' src/dsp/block_imp.c
grep -q 'hmx_native_u8_segmented_v4' src/host/block_main.c
python3 -m py_compile scripts/prepare_exp0161_long_delta_cache.py
python3 -m py_compile scripts/prepare_exp0161_segmented_cache.py
python3 -m py_compile scripts/summarize_exp0161_phase_b.py
printf 'EXP-0161 static gate: PASS\n'
