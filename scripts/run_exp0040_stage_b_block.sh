#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0040-stage-b}"
variant="${1:-W4U8}"
repeat_count="${2:-1}"
mlp_mode="${3:-w4u8_streaming}"
mlp_hvx_contexts="${4:-3}"
attribution_mode="${5:-on}"
audit_mode="${6:-off}"

if [[ "${DEPLOY_EXP0040_STAGE_B:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0040_stage_b_block.sh"
fi

if [[ "${variant}" == "W4U8" ]]; then
    common_ops_mode=scalar
    residual_mode=scalar
    f16_projection_mode=serial
    w4_pipeline_mode=control
    attention_pack_mode=control
    attention_pipeline_mode=control
    attention_hvx_contexts=1
    crouton_boundary_mode=control
    w4f16_hvx_workers=2
    w4f16_region_tiles=32
elif [[ "${variant}" == "W4F16" ]]; then
    common_ops_mode=hvx
    residual_mode=fused
    f16_projection_mode=serial
    w4_pipeline_mode=adaptive_down96_gate4_dma8_cross
    attention_pack_mode=hvx
    attention_pipeline_mode=gqa_qkv_overlap
    attention_hvx_contexts=4
    crouton_boundary_mode=norms
    w4f16_hvx_workers=3
    w4f16_region_tiles=32
    mlp_mode=crouton_native_batch8
    mlp_hvx_contexts=4
else
    printf 'unsupported Stage-B variant: %s\n' "${variant}" >&2
    exit 2
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${w4f16_hvx_workers} ${w4f16_region_tiles} ${common_ops_mode} ${attribution_mode} ${audit_mode} ${residual_mode} ${f16_projection_mode} ${w4_pipeline_mode} ${attention_pack_mode} ${mlp_mode} ${mlp_hvx_contexts} 64 ${attention_pipeline_mode} ${attention_hvx_contexts} ${crouton_boundary_mode}"
