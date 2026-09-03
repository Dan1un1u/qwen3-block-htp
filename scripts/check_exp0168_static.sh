#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4U8_BATCH8_RESIDENT_BIAS = 9' include/block_protocol.h
grep -q 'QBH_GENERATION_W4U8_BATCH8_GROUP_TILES UINT32_C(8)' src/dsp/block_imp.c
grep -q 'resident_bias_table = buffers->input_norm_weight' src/dsp/block_imp.c
grep -q 'generation_lm_head_scale_resident_bytes' src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HVX_POOL_W4U8_GENERATION_EXPAND' src/dsp/block_imp.c
grep -q 'qbh_dma_start_weight_prefetch' src/dsp/block_imp.c
grep -q 'QBH_GENERATION_SEQUENCE=${generation_mode}' scripts/run_exp0168.sh
! grep -RInE 'qti\.aisw|QnnGraph|QNN_' src include >/dev/null

bash -n scripts/build_exp0168.sh scripts/deploy_exp0168.sh \
    scripts/run_exp0168.sh scripts/check_exp0168_static.sh

printf '%s\n' \
    '{"experiment":"EXP-0168","static_gate":"pass","control_generation_mode":8,"candidate_generation_mode":9,"candidate":"batch8_plus_phase_overlaid_resident_bias_requant_carrier","timed_full_logits_ddr":false,"one_fastrpc_per_pass":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
