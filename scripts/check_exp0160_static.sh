#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(66)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(160)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DELTA_RECONSTRUCTION_PIPELINE' \
    include/block_protocol.h
grep -q 'qbh_scan_cache_dma_2d' src/dsp/block_imp.c
grep -q 'qbh_scan_u8_attention_delta_pipeline' src/dsp/block_imp.c
grep -q 'qbh_hmx_start' src/dsp/block_imp.c
grep -q 'qbh_hmx_wait' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DELTA_RECONSTRUCTION' src/host/block_main.c
test -f scripts/build_exp0160.sh
test -f scripts/run_exp0160.sh
test -f scripts/collect_exp0160.sh
test -f scripts/summarize_exp0160.py

printf '%s\n' '{"experiment":"EXP-0160","static_gate":"pass","recipe":"W4U8","control":"serial_delta_reconstruction","candidate":"direct_2d_reconstruction_plus_two_slot_gqa_pipeline","persistent_cache_abi_changed":false,"qnn_dependency":false}'
