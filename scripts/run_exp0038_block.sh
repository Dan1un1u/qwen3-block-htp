#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0022-block}"
variant="${1:-F16F16}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"
residual_mode="${5:-fused}"
f16f16_projection_mode="${6:-gate8}"
common_ops_mode="${7:-hvx}"
w4f16_hvx_workers="${8:-2}"
w4f16_region_tiles="${9:-32}"
w4f16_pipeline_mode="${10:-control}"
attention_pack_mode="${11:-hvx}"
mlp_mode="${12:-crouton_native_batch8}"
mlp_hvx_contexts="${13:-4}"
mlp_chunk_vectors="${14:-64}"
attention_pipeline_mode="${15:-gqa_qkv_overlap}"
attention_hvx_contexts="${16:-4}"
crouton_boundary_mode="${17:-control}"

if [[ "${DEPLOY_EXP0038:-0}" == "1" ]]; then
    "${project_root}/scripts/deploy_exp0022_block.sh"
fi

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell \
    "test -x ${remote_root}/qwen3_block_cli && test -f ${remote_root}/block_package_layer14_m64/manifest.json"
"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_block_cli ${remote_root}/block_package_layer14_m64 ${variant} ${repeat_count} ${w4f16_hvx_workers} ${w4f16_region_tiles} ${common_ops_mode} ${attribution_mode} ${audit_mode} ${residual_mode} ${f16f16_projection_mode} ${w4f16_pipeline_mode} ${attention_pack_mode} ${mlp_mode} ${mlp_hvx_contexts} ${mlp_chunk_vectors} ${attention_pipeline_mode} ${attention_hvx_contexts} ${crouton_boundary_mode}"
