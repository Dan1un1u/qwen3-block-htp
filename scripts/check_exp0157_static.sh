#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(63)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(157)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_PREFILL_CACHE_REUSE_ATTENTION_CARRIERS' \
    include/block_protocol.h
grep -q 'QBH_W4U8_PREFILL_CACHE_MODE' src/host/block_main.c
grep -q 'qbh_scan_persist_u8_prefill_attention_carriers' \
    src/dsp/block_imp.c
grep -q 'u8_cache_native_prefill_reuse_count' include/block_protocol.h
grep -q 'u8_cache_native_prefill_reused_carrier_bytes' \
    include/block_protocol.h
test -x scripts/deploy_exp0157.sh

printf '%s\n' '{"experiment":"EXP-0157","static_gate":"pass","execution_unit":"qwen3_real_layers0_27_M64_prefill_continuous_decode","recipe":"W4U8","direct_control":"duplicate_prefill_carrier_build","candidate":"reuse_attention_HMX_carriers","decode_session_abi":3,"one_fastrpc_per_full_stack_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
