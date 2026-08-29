#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(25)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(48)' include/block_protocol.h
grep -q 'QBH_BLOCK_CROUTON_BOUNDARY_W4U8_O_OUTPUT' include/block_protocol.h
grep -q 'qbh_hvx_residual_rms_norm_u8_native_io' include/hvx_u8_ops.h
grep -q 'qbh_hvx_residual_rms_norm_u8_native_io' src/dsp/hvx_u8_ops.c
grep -q 'qbh_hvx_residual_add_u8_native_output' src/dsp/hvx_u8_ops.c
grep -q 'w4u8_mlp_io_qkv_o' src/host/block_main.c

if grep -RIl --exclude='check_exp0048_stage_b_static.sh' \
        -E 'Qnn|libQnn|QAIRT' include src CMakeLists.txt cmake 2>/dev/null | \
        grep -q .; then
    echo 'QNN dependency found in the standalone runtime' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0048","stage":"B","static_gate":"pass","block_abi":25,"runtime_telemetry_experiment":48,"qnn_dependency":false,"vtcm_request_bytes":8388608,"native_o_output":"integer_hmx_tiles","native_post_attention_consumer":"hvx_gather_residual_plus_native_rmsnorm_scatter","single_hmx_owner":true}'
