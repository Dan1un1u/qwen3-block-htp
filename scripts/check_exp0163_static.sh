#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(70)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(163)' include/block_protocol.h
! grep -q 'experiment\\\":162' src/host/block_main.c
! grep -q 'EXP-0162' src/host/block_main.c
grep -q 'QBH_BLOCK_M + decode_steps' src/host/block_main.c
grep -q 'QBH_DECODE_SESSION_ABI_VERSION UINT32_C(4)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_HMX_U8_SEGMENT_COUNT(capacity_)' include/block_protocol.h
grep -q 'qbh_scan_append_u8_kv_hmx_segmented' src/dsp/block_imp.c
grep -q 'candidate_segmented_capacity257' scripts/deploy_exp0163.sh
grep -q 'QBH_KV_CACHE_CAPACITY=257' scripts/run_exp0163.sh
grep -q 'QBH_REPLAY_DECODE_STEPS=192' scripts/run_exp0163.sh
python3 -m py_compile \
    scripts/extend_exp0163_replay_source.py \
    scripts/generate_exp0162_exact_replay.py \
    scripts/prepare_exp0162_cache_packages.py \
    scripts/summarize_exp0163.py
printf '%s\n' '{"experiment":"EXP-0163","static_gate":"pass","layers":28,"prefill_rows":64,"decode_steps":192,"final_valid_length":256,"cache_capacity":257,"seal_positions":[95,127,159,191,223,255],"segment_tokens":32,"qnn_dependency":false}'
