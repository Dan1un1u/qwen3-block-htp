#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_mlp_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_executable}" "${host_stub}" "${dsp_skel}"; do
    [[ -f "${artifact}" ]] || {
        printf 'missing EXP-0040 build artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

if grep -RIn --exclude-dir=.git --exclude-dir=android_ReleaseG_aarch64 \
        --exclude-dir=hexagon_ReleaseG_toolv19_v79 \
        -E 'debug_(middle|gate|up)|debug_capture' \
        "${project_root}/include" "${project_root}/src"; then
    printf 'temporary EXP-0040 tensor capture remains in the source tree\n' >&2
    exit 1
fi

if grep -aEiq 'Qnn|QAIRT' \
        "${host_executable}" "${host_stub}" "${dsp_skel}"; then
    printf 'QNN or QAIRT dependency found in EXP-0040 binaries\n' >&2
    exit 1
fi

grep -q '#define QBH_MLP_ABI_VERSION UINT32_C(5)' \
    "${project_root}/include/mlp_protocol.h"
grep -q '#define QBH_EXPECTED_FULL_VTCM_BYTES UINT32_C(8388608)' \
    "${project_root}/include/probe_protocol.h"
grep -q '#define QBH_W4U8_VTCM_BYTES QBH_EXPECTED_FULL_VTCM_BYTES' \
    "${project_root}/include/probe_protocol.h"
grep -q 'qbh_mlp_gate_up_requant_lut_hvx' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'qbh_unpack_w4_to_s8_hvx' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"

printf '%s\n' '{"experiment":"EXP-0040","static_gate":"pass","mlp_abi":5,"qnn_dependency":false,"temporary_tensor_capture":false,"vtcm_request_bytes":8388608}'
