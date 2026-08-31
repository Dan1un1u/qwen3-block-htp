#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(44)' "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(108)' "${project_root}/include/block_protocol.h"
grep -q 'qkv_norms' "${project_root}/src/host/block_main.c"
grep -q 'qbh_projection_direct_qkv_crouton' "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_audit_crouton_qkv_operands' "${project_root}/src/dsp/block_imp.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0108","static_gate":"pass","qkv_norms_runtime":true,"existing_crouton_qkv_carrier":true,"qnn_dependency":false}'
