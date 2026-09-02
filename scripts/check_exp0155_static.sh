#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(61)' include/block_protocol.h
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(155)' include/block_protocol.h
grep -q 'QBH_DECODE_SESSION_ABI_VERSION UINT32_C(3)' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_K_WEIGHT_V1' include/block_protocol.h
grep -q 'QBH_KV_CACHE_FORMAT_HMX_U8_V_WEIGHT_V1' include/block_protocol.h
grep -q 'qbh_scan_append_u8_kv_hmx_native' src/dsp/block_imp.c
grep -q 'qbh_attention_u8_update_k_native_token' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_update_v_native_token' src/dsp/attention_u8_core.c
grep -q 'QBH_KV_CACHE_LAYOUT' src/host/block_main.c
grep -q 'u8_cache_full_prefix_pack_count' src/host/block_main.c
grep -q 'QBH_VERTICAL_SLICE_FIRST_LAYER QBH_REPLAY_LAYER_INDEX' \
    include/block_protocol.h
grep -q 'QBH_VERTICAL_SLICE_LAYER_COUNT UINT32_C(1)' \
    include/block_protocol.h

python_bin="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
"${python_bin}" -m py_compile scripts/prepare_exp0155_hmx_cache.py
"${python_bin}" -m py_compile scripts/prepare_exp0155_logical_reference.py
printf 'EXP-0155 static gate passed\n'
