#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(65)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(159)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_DELTA_V2' \
    include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_DELTA_V2' \
    include/block_protocol.h
grep -q 'qbh_scan_append_u8_kv_hmx_delta' src/dsp/block_imp.c
grep -q 'hmx_native_u8_delta' src/host/block_main.c
test -x scripts/build_exp0159.sh
test -x scripts/deploy_exp0159.sh
test -x scripts/run_exp0159.sh
test -x scripts/prepare_exp0159_u8_delta_cache_package.py
test -x scripts/collect_exp0159.sh
test -x scripts/summarize_exp0159.py

printf '%s\n' '{"experiment":"EXP-0159","static_gate":"pass","recipe":"W4U8","direct_control":"HMX_U8_WEIGHT_V1_DDR_tile_RMW","candidate":"HMX_U8_WEIGHT_DELTA_V2_immutable_carrier_plus_contiguous_journal","prefill_tokens":64,"decode_steps":8,"one_fastrpc_per_full_stack_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"qnn_dependency":false}'
