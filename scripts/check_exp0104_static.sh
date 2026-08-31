#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools_root="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin"
skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
candidate_symbol="qbh_qk_norm_rope_two_heads_u8_quarter_tiled"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(104)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_QK_PAIR_QUARTER_TILED' \
    "${project_root}/src/dsp/hvx_u8_ops.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi

candidate_line="$(${tools_root}/hexagon-nm -S "${skel}" | grep " ${candidate_symbol}$")"
candidate_start="0x$(awk '{print $1}' <<<"${candidate_line}")"
candidate_size="$((16#$(awk '{print $2}' <<<"${candidate_line}")))"
candidate_stop="$((candidate_start + candidate_size))"
disassembly="$(${tools_root}/hexagon-llvm-objdump -d \
    --start-address="${candidate_start}" --stop-address="${candidate_stop}" \
    "${skel}")"
if grep -Eq 'vmem\(r29|vmem\(r30' <<<"${disassembly}"; then
    printf 'candidate helper contains a vector stack access\n' >&2
    exit 1
fi
if ! grep -q 'vdeal' <<<"${disassembly}"; then
    printf 'candidate helper does not contain quarter-tile vdeal\n' >&2
    exit 1
fi
if grep -q 'allocframe(#0x18)' <<<"${disassembly}"; then
    scalar_frame_bytes=24
else
    scalar_frame_bytes=0
fi
printf '{"experiment":"EXP-0104","static_gate":"pass","candidate_symbol_bytes":%u,"candidate_scalar_abi_frame_bytes":%u,"candidate_vector_stack_access":false,"candidate_vdeal":true,"qnn_dependency":false}\n' \
    "${candidate_size}" "${scalar_frame_bytes}"
