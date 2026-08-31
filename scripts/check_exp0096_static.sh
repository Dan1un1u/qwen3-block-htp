#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump="${HEXAGON_OBJDUMP:-/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump}"
object="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(96)' \
    "${project_root}/include/block_protocol.h"
grep -q 'PERSISTENT_MLP_HVX_ARITHMETIC_ACTIVATION = 8' \
    "${project_root}/include/block_protocol.h"
"${objdump}" -d --disassemble-symbols=qbh_mlp_gate_up_hvx \
    "${object}" > "${temporary}/candidate.disasm"
"${objdump}" -d --disassemble-symbols=qbh_mlp_gate_up_lut_hvx \
    "${object}" > "${temporary}/control.disasm"

candidate_gather="$(grep -c 'vgather' "${temporary}/candidate.disasm" || true)"
candidate_multiply="$(grep -c 'vmpy' "${temporary}/candidate.disasm" || true)"
candidate_shift_pack="$(grep -Ec 'vasr|vpack' \
    "${temporary}/candidate.disasm" || true)"
candidate_vector_stack="$(grep -Ec 'vmem\(r29|vmem\(r30' \
    "${temporary}/candidate.disasm" || true)"
control_gather="$(grep -c 'vgather' "${temporary}/control.disasm" || true)"
if [[ "${candidate_gather}" -ne 0 || "${candidate_multiply}" -lt 4 || \
      "${candidate_shift_pack}" -lt 3 || "${candidate_vector_stack}" -ne 0 || \
      "${control_gather}" -lt 4 ]]; then
    printf '{"experiment":"EXP-0096","static_gate":"fail","candidate_vgather":%s,"candidate_vmpy":%s,"candidate_shift_pack":%s,"candidate_vector_stack_refs":%s,"control_vgather":%s}\n' \
        "${candidate_gather}" "${candidate_multiply}" \
        "${candidate_shift_pack}" "${candidate_vector_stack}" \
        "${control_gather}"
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0096","static_gate":"pass","candidate_vgather":0,"candidate_vmpy":%s,"candidate_shift_pack":%s,"candidate_vector_stack_refs":0,"control_vgather":%s,"qnn_dependency":false}\n' \
    "${candidate_multiply}" "${candidate_shift_pack}" "${control_gather}"
