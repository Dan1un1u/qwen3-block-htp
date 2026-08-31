#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools_root="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin"
skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
symbols=(
    qbh_attention_u8_log2_codes_paired_dual_head_reduction
    qbh_attention_u8_sum_log2_weights_dual_head
    qbh_attention_u8_sum_probability_dual_head
)
total_bytes=0

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(105)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_W4U8_SOFTMAX_REDUCTION_DUAL_HEAD' \
    "${project_root}/src/dsp/attention_u8_core.c"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi

for symbol in "${symbols[@]}"; do
    line="$(${tools_root}/hexagon-nm -S "${skel}" | grep " ${symbol}$")"
    start="0x$(awk '{print $1}' <<<"${line}")"
    size="$((16#$(awk '{print $2}' <<<"${line}")))"
    stop="$((start + size))"
    disassembly="$(${tools_root}/hexagon-llvm-objdump -d \
        --start-address="${start}" --stop-address="${stop}" "${skel}")"
    if grep -Eq 'vmem\(r29|vmem\(r30' <<<"${disassembly}"; then
        printf '%s contains a vector stack access\n' "${symbol}" >&2
        exit 1
    fi
    total_bytes="$((total_bytes + size))"
done

printf '{"experiment":"EXP-0105","static_gate":"pass","candidate_symbol_count":3,"candidate_symbol_bytes":%u,"candidate_vector_stack_access":false,"carrier_layout_changed":false,"qnn_dependency":false}\n' \
    "${total_bytes}"
