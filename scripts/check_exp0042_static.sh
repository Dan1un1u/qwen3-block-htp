#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    [[ -f "${artifact}" ]] || exit 1
done
if grep -aEiq 'Qnn|QAIRT' \
        "${host_executable}" "${host_stub}" "${dsp_skel}"; then
    printf 'QNN or QAIRT dependency found in EXP-0042 binaries\n' >&2
    exit 1
fi
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(22)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(42)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)' \
    "${project_root}/include/probe_protocol.h"
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA' \
    "${project_root}/include/block_protocol.h"
grep -q 'qbh_attention_u8_softmax_group' \
    "${project_root}/src/dsp/attention_u8_core.c"
grep -q 'Q6_Vb_vlut32_VbVbR_nomatch' \
    "${project_root}/src/dsp/attention_u8_core.c"
grep -q 'qbh_hvx_qk_norm_rope_u8_native_head' \
    "${project_root}/src/dsp/hvx_u8_ops.c"

printf '%s\n' '{"experiment":"EXP-0042","static_gate":"pass","block_abi":22,"qnn_dependency":false,"vtcm_request_bytes":8388608,"integer_attention":true}'
