#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_GENERATION_DEFAULT_TOKENS UINT32_C(16)' \
    include/block_protocol.h
grep -q '#define QBH_GENERATION_MAX_TOKENS UINT32_C(193)' \
    include/block_protocol.h
grep -q 'QBH_EXP0169_GENERATION_STEPS:-193' scripts/run_exp0169.sh
grep -q 'QBH_KV_CACHE_CAPACITY=257' scripts/run_exp0169.sh
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS' \
    src/dsp/block_imp.c
grep -q 'generation_steps > QBH_GENERATION_DEFAULT_TOKENS' \
    src/host/block_main.c
bash -n scripts/build_exp0169.sh scripts/deploy_exp0169.sh \
    scripts/run_exp0169.sh scripts/run_exp0169_formal.sh \
    scripts/check_exp0169_static.sh
python3 -m py_compile scripts/prepare_exp0169_long_generation_overlay.py \
    scripts/summarize_exp0169.py scripts/verify_exp0167_generation.py

printf '%s\n' '{"experiment":"EXP-0169","static_gate":"pass","variant":"W4U8","source_parent":"EXP-0168","prefill_tokens":64,"decode_steps":192,"final_valid_length":256,"kernel_changes":false,"qnn_dependency":false}'
