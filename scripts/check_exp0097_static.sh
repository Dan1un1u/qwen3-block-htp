#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${QBH_EXP0097_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
objdump="${HEXAGON_OBJDUMP:-/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump}"
object="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(97)' \
    "${project_root}/include/block_protocol.h"
grep -q 'PERSISTENT_MLP_HVX_EXACT_AFFINE = 8' \
    "${project_root}/include/block_protocol.h"
"${project_root}/scripts/check_exp0097_affine_math.py" \
    "${project_root}/src/dsp/mlp_u8.c" \
    "${package}/silu_up_lut_u16.bin" > "${temporary}/math.json"
"${objdump}" -d --disassemble-symbols=qbh_mlp_gate_up_exact_affine_hvx \
    "${object}" > "${temporary}/candidate.disasm"
"${objdump}" -d --disassemble-symbols=qbh_mlp_gate_up_lut_hvx \
    "${object}" > "${temporary}/control.disasm"

candidate_gather="$(grep -c 'vgather' "${temporary}/candidate.disasm" || true)"
candidate_multiply="$(grep -c 'vmpy' "${temporary}/candidate.disasm" || true)"
candidate_shift="$(grep -c 'vasr' "${temporary}/candidate.disasm" || true)"
candidate_pack="$(grep -c 'vpack' "${temporary}/candidate.disasm" || true)"
candidate_vector_stack="$(grep -Ec 'vmem\(r29|vmem\(r30' \
    "${temporary}/candidate.disasm" || true)"
control_gather="$(grep -c 'vgather' "${temporary}/control.disasm" || true)"
if [[ "${candidate_gather}" -ne 2 || "${candidate_multiply}" -lt 2 || \
      "${candidate_shift}" -lt 4 || "${candidate_pack}" -lt 1 || \
      "${candidate_vector_stack}" -ne 0 || "${control_gather}" -lt 4 ]]; then
    printf '{"experiment":"EXP-0097","static_gate":"fail","candidate_vgather":%s,"candidate_vmpy":%s,"candidate_vasr":%s,"candidate_vpack":%s,"candidate_vector_stack_refs":%s,"control_vgather":%s}\n' \
        "${candidate_gather}" "${candidate_multiply}" \
        "${candidate_shift}" "${candidate_pack}" \
        "${candidate_vector_stack}" "${control_gather}"
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0097","static_gate":"pass","math_gate":"pass_65536_of_65536","coefficient_bytes":512,"candidate_vgather":2,"candidate_vmpy":%s,"candidate_vasr":%s,"candidate_vpack":%s,"candidate_vector_stack_refs":0,"control_vgather":%s,"qnn_dependency":false}\n' \
    "${candidate_multiply}" "${candidate_shift}" \
    "${candidate_pack}" "${control_gather}"
