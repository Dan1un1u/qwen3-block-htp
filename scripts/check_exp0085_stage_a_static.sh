#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
protocol="${project_root}/include/block_protocol.h"
dsp="${project_root}/src/dsp/block_imp.c"
host="${project_root}/src/host/block_main.c"

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(45)' "${protocol}"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(85)' "${protocol}"
grep -q 'qkv_schedule_mode' "${protocol}"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_GQA_GROUP_MAJOR' "${protocol}"
grep -q 'QBH_BLOCK_QKV_SCHEDULE_Q_PREFIX4_K_ALL' "${protocol}"
grep -q 'qkv_timeline_q_projection_ready' "${protocol}"
grep -q 'qbh_qkv_timeline_record_projection' "${dsp}"
grep -q 'qbh_qkv_timeline_record_prep_task' "${dsp}"
grep -q 'qbh_qkv_timeline_record_attention' "${dsp}"
grep -q 'qbh_run_f16f16_qkv_group_major' "${dsp}"
grep -q 'qbh_run_w4f16_qkv_group_major' "${dsp}"
grep -q 'qbh_run_w4u8_qkv_group_major' "${dsp}"
grep -q 'QBH_DUMP_QKV_TIMELINE_PATH' "${host}"
grep -q 'QBH_QKV_TIMELINE' "${host}"
grep -q 'QBH_QKV_SCHEDULE' "${host}"
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in formal source\n' >&2
    exit 1
fi
printf '%s\n' '{"experiment":"EXP-0085","stage":"A/B","static_gate":"pass","runtime_schedule_selectable":true,"qnn_dependency":false}'
