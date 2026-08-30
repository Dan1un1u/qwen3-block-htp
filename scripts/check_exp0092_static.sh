#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump="${HEXAGON_OBJDUMP:-/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump}"
skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT

"${objdump}" -d \
    --disassemble-symbols=qbh_unpack_w4_to_s8_hvx,qbh_unpack_w4_to_s8_hvx_interleaved2 \
    "${skel}" > "${temporary}/decoder.disasm"
awk '/<qbh_unpack_w4_to_s8_hvx>:/ {copy=1} /<qbh_unpack_w4_to_s8_hvx_interleaved2>:/ {copy=0} copy' \
    "${temporary}/decoder.disasm" > "${temporary}/control.disasm"
awk '/<qbh_unpack_w4_to_s8_hvx_interleaved2>:/ {copy=1} copy' \
    "${temporary}/decoder.disasm" > "${temporary}/candidate.disasm"
control_lut="$(grep -c 'vlut32' "${temporary}/control.disasm")"
control_loads="$(grep -c 'vmem(r0++' "${temporary}/control.disasm")"
candidate_lut="$(grep -c 'vlut32' "${temporary}/candidate.disasm")"
candidate_loads="$(grep -c 'vmem(r0++' "${temporary}/candidate.disasm")"
candidate_stack="$(grep -Ec 'r29|sp\)' "${temporary}/candidate.disasm" || true)"
if [[ "${control_lut}" -ne 2 || "${control_loads}" -ne 1 || \
      "${candidate_lut}" -ne 4 || "${candidate_loads}" -ne 2 || \
      "${candidate_stack}" -ne 0 ]]; then
    printf '{"static_gate":"fail","control_vlut32":%s,"control_loads":%s,"candidate_vlut32":%s,"candidate_loads":%s,"candidate_stack_refs":%s}\n' \
        "${control_lut}" "${control_loads}" "${candidate_lut}" \
        "${candidate_loads}" "${candidate_stack}"
    exit 1
fi
printf '{"static_gate":"pass","control_vectors_per_loop":1,"candidate_vectors_per_loop":2,"candidate_stack_refs":0}\n'
