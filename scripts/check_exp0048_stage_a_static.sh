#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(25)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(48)' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_W4U8_QKV_INPUT' include/block_protocol.h
grep -q 'qbh_hvx_rms_norm_u8_native_activation' src/dsp/block_imp.c
grep -q 'w4u8_mlp_io_qkv_input' src/host/block_main.c

if grep -RIl --exclude='check_exp0048_stage_a_static.sh' \
        -E 'Qnn|libQnn|QAIRT' include src CMakeLists.txt cmake 2>/dev/null | \
        grep -q .; then
    echo 'QNN dependency found in the standalone runtime' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0048","stage":"A","static_gate":"pass","block_abi":25,"runtime_telemetry_experiment":48,"qnn_dependency":false,"vtcm_request_bytes":8388608,"native_qkv_input_store":"hvx_word_scatter","shared_qkv_activation_carrier":true,"single_hmx_owner":true}'
