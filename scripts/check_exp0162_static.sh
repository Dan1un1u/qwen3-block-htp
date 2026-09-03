#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(70)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(162)' include/block_protocol.h
grep -q 'QBH_DECODE_SESSION_ABI_VERSION UINT32_C(4)' include/block_protocol.h
grep -q 'u8_cache_segment_seal_count' include/block_protocol.h
grep -q 'qbh_scan_append_u8_kv_hmx_segmented' src/dsp/block_imp.c
grep -q 'max_segments \* QBH_KV_CACHE_HMX_U8_SEGMENT_K_BYTES' \
    src/dsp/block_imp.c
grep -q 'header->kv_cache_capacity - QBH_BLOCK_M' src/host/block_main.c
grep -q 'candidate_segmented_capacity104' scripts/deploy_exp0162.sh
python3 -m py_compile \
    scripts/export_exp0148_replay.py \
    scripts/export_exp0149_vertical_slice.py \
    scripts/generate_exp0162_exact_replay.py \
    scripts/prepare_exp0162_cache_packages.py
printf '%s\n' '{"experiment":"EXP-0162","static_gate":"pass","layers":28,"prefill_rows":64,"decode_steps":40,"cache_capacity":104,"segment_tokens":32,"qnn_dependency":false}'
