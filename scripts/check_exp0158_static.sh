#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(64)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(158)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_F16_K_WEIGHT_V1' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_F16_V_WEIGHT_V1' include/block_protocol.h
grep -q 'qbh_scan_persist_f16_prefill_attention_carriers' src/dsp/block_imp.c
grep -q 'qbh_scan_append_f16_kv_hmx_native' src/dsp/block_imp.c
grep -q 'f16_cache_full_prefix_pack_count' src/dsp/block_imp.c
! grep -RInE 'qti\.aisw|QnnGraph|QNN_' src include >/dev/null

printf '%s\n' '{"experiment":"EXP-0158","static_gate":"pass","recipes":["F16F16","W4F16"],"control":"head_major_row_v1","candidate":"HMX_F16_K_WEIGHT_V1/HMX_F16_V_WEIGHT_V1","prefill":"reuse_exact_m64_attention_carriers","decode":"contiguous_delta_journal_then_hvx_tile_patch","one_fastrpc_per_full_stack_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
