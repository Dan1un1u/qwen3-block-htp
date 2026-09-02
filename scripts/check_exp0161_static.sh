#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(67)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(161)' include/block_protocol.h
grep -q 'reference_kv_cache_k_hmx_%s.bin' src/host/block_main.c
grep -q 'reference_kv_cache_v_hmx_%s.bin' src/host/block_main.c
grep -q 'qbh_scan_u8_attention_delta_pipeline' src/dsp/block_imp.c
python3 -m py_compile scripts/prepare_exp0161_long_delta_cache.py
printf 'EXP-0161 static gate: PASS\n'
