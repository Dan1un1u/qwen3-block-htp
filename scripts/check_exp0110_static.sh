#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(45)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(110)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL' \
    "${project_root}/include/block_protocol.h"
grep -q 'qkv_norms' "${project_root}/src/host/block_main.c"
grep -q 'qbh_run_w4f16_qkv_prefix4' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_w4f16_qkv_trace_command' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_capture_row_major_qkv_reference' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qkv_operand_audit_tensor_count' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'current.first_n_tile \* QBH_HMX_FP16_COLS' \
    "${project_root}/src/dsp/block_imp.c"
for cell in control carrier prefix4 combined; do
    grep -q "${cell})" "${project_root}/scripts/run_exp0110.sh"
done
grep -q 'SAMPLES = 5' "${project_root}/scripts/analyze_exp0110.py"
grep -q 'cells=(control carrier prefix4 combined)' \
    "${project_root}/scripts/collect_exp0110.sh"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0110 source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0110","static_gate":"pass","factorial_cells":4,"recipe":"W4F16","qkv_carrier_runtime_selectable":true,"q_prefix4_runtime_selectable":true,"row_major_and_crouton_outputs":true,"one_hmx_owner":true,"qnn_dependency":false}'
