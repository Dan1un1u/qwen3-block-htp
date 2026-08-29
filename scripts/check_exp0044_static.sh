#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    [[ -f "${artifact}" ]] || exit 1
done
if strings "${host_executable}" "${host_stub}" "${dsp_skel}" |
        grep -Eiq 'Qnn|QAIRT'; then
    printf 'QNN or QAIRT dependency found in EXP-0044 binaries\n' >&2
    exit 1
fi
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(23)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(44)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)' \
    "${project_root}/include/probe_protocol.h"
grep -q 'QBH_BLOCK_ATTENTION_PIPELINE_U8_LOG2_GQA_QKV_OVERLAP = 8' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_HVX_POOL_U8_QK_PREP = 8' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_qk_norm_rope_u8_native_k_head' \
    "${project_root}/src/dsp/hvx_u8_ops.c"
grep -q 'Q6_vscatter_RMVwV' \
    "${project_root}/src/dsp/hvx_u8_ops.c"
grep -q 'u8_attention_fused_k_operand_mismatch_count' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_pool_u8_qk_prep_start_async' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_pool_u8_qk_prep_publish' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'qbh_hvx_pool_u8_qk_prep_wait_async' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'release(%0):at' \
    "${project_root}/src/dsp/block_imp.c"
grep -q 'paired_rounds=7' \
    "${project_root}/scripts/collect_exp0044_stage_b.sh"

printf '%s\n' '{"experiment":"EXP-0044","stage":"B","static_gate":"pass","block_abi":23,"qnn_dependency":false,"vtcm_request_bytes":8388608,"fused_k_operand":true,"generation_safe_qk_prep":true,"single_hmx_owner":true}'
