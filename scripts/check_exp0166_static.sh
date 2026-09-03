#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${1:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/w4f16_greedy16}"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(72)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(166)' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_LM_HEAD_OVERLAP' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_DMA_HVX_OVERLAP' include/block_protocol.h
grep -q 'QBH_BLOCK_GENERATION_GREEDY_W4F16_COARSE_PIPELINE' include/block_protocol.h
grep -q 'qbh_run_generation_head_w4f16_overlap' src/dsp/block_imp.c
grep -q 'qbh_hmx_start_fp16_tile_scales_streaming' src/dsp/block_imp.c
grep -q 'generation_lm_head_scale_resident_bytes' src/dsp/block_imp.c
grep -q 'QBH_GENERATION_SEQUENCE=${generation_mode}' scripts/run_exp0166.sh
! grep -RInE 'qti\.aisw|QnnGraph|QNN_' src include >/dev/null

bash -n scripts/build_exp0166.sh scripts/deploy_exp0166.sh \
    scripts/run_exp0166.sh

test "$(stat -c %s "${package}/generation_lm_head_weight_w4_hmx.bin")" \
    -eq 155582464
test "$(stat -c %s "${package}/generation_lm_head_weight_w4_scale_f32.bin")" \
    -eq 607744

printf '%s\n' \
    '{"experiment":"EXP-0166","static_gate":"pass","control_generation_mode":4,"streaming_hmx_mode":5,"dma_hvx_overlap_mode":6,"coarse_pipeline_mode":7,"candidate":"resident_scales_plus_compressed_DMA_ping_pong_plus_coarse_HVX_HMX_pipeline","timed_full_logits_ddr":false,"one_fastrpc_per_pass":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
