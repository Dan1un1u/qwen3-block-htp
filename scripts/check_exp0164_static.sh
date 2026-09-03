#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${1:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/w4f16_greedy16}"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(71)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(164)' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16' include/block_protocol.h
grep -q 'qbh_stage_generation_embedding' src/dsp/block_imp.c
grep -q 'qbh_run_generation_head_w4f16' src/dsp/block_imp.c
grep -q 'qbh_pack_fp16_activation_row0' src/dsp/block_imp.c
grep -q 'qbh_run_generation_sequence' src/host/block_main.c
grep -q 'QBH_GENERATION_SEQUENCE=1' scripts/run_exp0164.sh
! grep -RInE 'qti\.aisw|QnnGraph|QNN_' src include >/dev/null

python3 -m py_compile \
    scripts/run_exp0164_semantic_gate.py \
    scripts/prepare_exp0164_generation_package.py \
    scripts/verify_exp0164_generation.py
bash -n \
    scripts/build_exp0164.sh \
    scripts/deploy_exp0164.sh \
    scripts/run_exp0164.sh

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
    '{"experiment":"EXP-0164","static_gate":"pass","variant":"W4F16","layers":28,"prompt_tokens":64,"generated_tokens":16,"vocab_size":151936,"lm_head":"streamed_per_output_channel_W4_to_FP16_HMX_online_argmax","timed_full_logits_ddr":false,"one_fastrpc_per_pass":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
