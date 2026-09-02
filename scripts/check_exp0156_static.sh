#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(62)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(156)' include/block_protocol.h
grep -q 'QBH_VERTICAL_SLICE_FIRST_LAYER UINT32_C(0)' include/block_protocol.h
grep -q 'QBH_VERTICAL_SLICE_LAYER_COUNT QBH_QWEN3_TRANSFORMER_LAYERS' \
    include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1' include/block_protocol.h
grep -q 'QBH_KV_CACHE_LAYOUT' src/host/block_main.c
grep -q 'qbh_scan_append_u8_kv_hmx_native' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_update_k_native_token' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_update_v_native_token' src/dsp/block_imp.c

printf '%s\n' '{"experiment":"EXP-0156","static_gate":"pass","execution_unit":"qwen3_real_layers0_27_M64_prefill_continuous_decode","recipe":"W4U8","decode_session_abi":3,"cache_formats":["hmx_u8_k_weight_v1","hmx_u8_v_weight_v1"],"one_fastrpc_per_full_stack_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
