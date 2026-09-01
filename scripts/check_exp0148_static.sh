#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_cli="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_cli}" "${host_stub}" "${dsp_skel}"; do
    test -f "${artifact}"
done
grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(55)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(148)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_DECODE_SESSION_ABI_VERSION UINT32_C(1)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_REPLAY_LAYER_INDEX UINT32_C(14)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_QWEN3_TRANSFORMER_LAYERS UINT32_C(28)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_REPLAY_DECODE_STEPS UINT32_C(8)' \
    "${project_root}/src/host/block_main.c"
grep -q 'replay_profile' \
    "${project_root}/src/host/block_main.c"

if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0148 source\n' >&2
    exit 1
fi
if readelf -d "${host_cli}" "${host_stub}" 2>/dev/null \
        | grep -qi 'libQnn'; then
    printf 'QNN dependency found in EXP-0148 Host artifacts\n' >&2
    exit 1
fi

printf '%s\n' '{"experiment":"EXP-0148","static_gate":"pass","execution_unit":"qwen3_layer14_real_replay_prefill_continuous_decode","decode_session_abi":1,"active_layer":14,"declared_layer_count":28,"prefill_positions":"0-63","decode_positions":"64-71","cache_capacity":72,"one_fastrpc_per_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"persistent_kv_cache_ddr_allowed":true,"qnn_dependency":false}'
