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
    printf 'QNN or QAIRT dependency found in EXP-0043 binaries\n' >&2
    exit 1
fi
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(23)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(43)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)' \
    "${project_root}/include/probe_protocol.h"
for field in w4u8_mlp_boundary_ticks w4u8_mlp_input_pack_ticks \
             w4u8_mlp_output_unpack_ticks w4u8_mlp_control_ticks; do
    grep -q "${field}" "${project_root}/include/block_protocol.h"
    grep -q "${field}" "${project_root}/src/dsp/block_imp.c"
    grep -q "${field}" "${project_root}/src/host/block_main.c"
done
grep -q 'paired_rounds=11' \
    "${project_root}/scripts/collect_exp0043_evidence.sh"

printf '%s\n' '{"experiment":"EXP-0043","static_gate":"pass","block_abi":23,"qnn_dependency":false,"vtcm_request_bytes":8388608,"optimization_changes_allowed":false,"exclusive_w4u8_mlp_boundary_ledger":true}'
