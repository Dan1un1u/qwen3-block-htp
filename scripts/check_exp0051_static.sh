#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(27)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(51)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_Q_HEAD_PAIR = 4' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_QKV_HEAD_PAIR = 5' include/block_protocol.h
grep -q 'QBH_BLOCK_HMX_U8S8_HEAD_PAIR = 7' src/dsp/block_imp.c
grep -q 'qbh_hmx_start_qkv_head_pair' src/dsp/block_imp.c
grep -q 'qbh_hvx_pool_u8_qk_prep_publish' src/dsp/block_imp.c
grep -q 'compressed_weight_third' src/dsp/block_imp.c
grep -q 'expanded_weight_third' src/dsp/block_imp.c

if grep -RIl --exclude='check_exp0051_static.sh' \
        -E 'Qnn|libQnn|QAIRT' include src CMakeLists.txt cmake 2>/dev/null | \
        grep -q .; then
    echo 'QNN dependency found in the standalone runtime' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0051","static_gate":"pass","block_abi":27,"runtime_telemetry_experiment":51,"qkv_control_commands_per_block":32,"q_only_candidate_commands_per_block":24,"full_qkv_candidate_commands_per_block":16,"heads_per_worker_command":2,"head_readiness_granularity":1,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
