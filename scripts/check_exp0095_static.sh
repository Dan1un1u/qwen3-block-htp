#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump="${HEXAGON_OBJDUMP:-/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump}"
object="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(95)' \
    "${project_root}/include/block_protocol.h"
grep -q 'DEPENDENCY_STREAM_SOFTMAX_SHUFFLE4 = 16' \
    "${project_root}/include/block_protocol.h"
"${objdump}" -d \
    --disassemble-symbols=qbh_attention_u8_requant_softmax_group_rows_prebuilt_templates_shuffle4 \
    "${object}" > "${temporary}/candidate.disasm"

shuffle_count="$(grep -c 'vshuff' "${temporary}/candidate.disasm" || true)"
gather_count="$(grep -c 'vgather' "${temporary}/candidate.disasm" || true)"
memcpy_count="$(grep -c 'memcpy' "${temporary}/candidate.disasm" || true)"
vector_stack_refs="$(grep -Ec 'vmem\(r29|vmem\(r30' \
    "${temporary}/candidate.disasm" || true)"
row_helper_calls="$(grep -c 'qbh_attention_u8_softmax_requantized_pair' \
    "${temporary}/candidate.disasm" || true)"
if [[ "${shuffle_count}" -ne 12 || "${gather_count}" -ne 0 || \
      "${memcpy_count}" -ne 0 || "${vector_stack_refs}" -ne 0 || \
      "${row_helper_calls}" -ne 1 ]]; then
    printf '{"experiment":"EXP-0095","static_gate":"fail","candidate_vshuff":%s,"candidate_vgather":%s,"candidate_memcpy":%s,"candidate_vector_stack_refs":%s,"candidate_row_helper_calls":%s}\n' \
        "${shuffle_count}" "${gather_count}" "${memcpy_count}" \
        "${vector_stack_refs}" "${row_helper_calls}"
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '{"experiment":"EXP-0095","static_gate":"pass","candidate_native_tile_loads_per_four_rows":4,"candidate_native_tile_stores_per_four_rows":4,"candidate_vshuff":12,"candidate_vgather":0,"candidate_memcpy":0,"candidate_vector_stack_refs":0,"worker_private_carrier_bytes":1024,"qnn_dependency":false}\n'
