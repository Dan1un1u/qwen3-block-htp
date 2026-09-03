#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${1:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/w4f16_greedy16}"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(71)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(165)' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH4' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_HVX_ARGMAX_BATCH8' include/block_protocol.h
grep -q 'qbh_hvx_reduce_max_f16' src/dsp/block_imp.c
grep -q 'Q6_W_vdeal_VVR' src/dsp/block_imp.c
grep -q 'QBH_GENERATION_SEQUENCE=${generation_mode}' scripts/run_exp0165.sh
! grep -RInE 'qti\.aisw|QnnGraph|QNN_' src include >/dev/null

bash -n \
    scripts/build_exp0165.sh \
    scripts/deploy_exp0165.sh \
    scripts/run_exp0165.sh \
    scripts/run_exp0165_formal.sh
python3 -m py_compile scripts/summarize_exp0165.py

test "$(stat -c %s "${package}/generation_prompt_token_ids_u32.bin")" \
    -eq 256
test "$(stat -c %s "${package}/generation_embedding_weight_f16.bin")" \
    -eq 622329856
test "$(stat -c %s "${package}/generation_final_norm_weight_f16.bin")" \
    -eq 4096
test "$(stat -c %s "${package}/generation_lm_head_weight_w4_hmx.bin")" \
    -eq 155582464
test "$(stat -c %s "${package}/generation_lm_head_weight_w4_scale_f32.bin")" \
    -eq 607744
test "$(stat -c %s "${package}/generation_expected_token_ids_u32.bin")" \
    -eq 64

printf '%s\n' \
    '{"experiment":"EXP-0165","static_gate":"pass","control_generation_mode":1,"diagnostic_generation_modes":[2,3],"candidate_generation_mode":4,"candidate":"HVX_group_max_with_rare_stable_scalar_lane_resolution_and_phase_overlaid_batch8","timed_full_logits_ddr":false,"one_fastrpc_per_pass":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
