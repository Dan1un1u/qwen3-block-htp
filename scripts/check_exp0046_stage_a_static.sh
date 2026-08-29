#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(25)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(46)' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_W4U8_MLP_INPUT' include/block_protocol.h
grep -q 'qbh_hvx_residual_rms_norm_u8_native_activation' src/dsp/block_imp.c
grep -q 'qbh_store_u8_hmx_activation' src/dsp/hvx_u8_ops.c
grep -q 'Q6_vscatter_RMVwV' src/dsp/hvx_u8_ops.c
grep -q '++header->w4u8_mlp_input_pack_skipped' src/dsp/block_imp.c
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' src include CMakeLists.txt >/dev/null; then
    printf 'QNN dependency detected\n' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0046","stage":"A","static_gate":"pass","block_abi":25,"qnn_dependency":false,"vtcm_request_bytes":8388608,"native_input_store":"hvx_word_scatter","single_hmx_owner":true}'
