#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
set -u

static_output_dir="${QBH_STATIC_OUTPUT_DIR:-$(mktemp -d)}"
owns_output_dir=0
if [[ -z "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    owns_output_dir=1
fi
trap 'if [[ "${owns_output_dir}" == "1" ]]; then rm -rf -- "${static_output_dir}"; fi' EXIT

mkdir -p "${static_output_dir}"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
host_cli="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"

"${hexagon_readelf}" -Ws "${dsp_skel}" \
    > "${static_output_dir}/exp0027.symbols.txt"
strings "${host_cli}" > "${static_output_dir}/exp0027.host.strings.txt"

grep -q 'EXP-0027' "${static_output_dir}/exp0027.host.strings.txt"
grep -q 'f16f16_projection_mode' \
    "${static_output_dir}/exp0027.host.strings.txt"
grep -q 'double_buffer_batch2' \
    "${static_output_dir}/exp0027.host.strings.txt"
grep -q 'qbh_run_f16f16_pipelined_projection' \
    "${static_output_dir}/exp0027.symbols.txt"

printf '{"experiment":"EXP-0027","f16f16_async_dma_hmx":true,'
printf '"f16f16_batch2":true,"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"qnn_dependency":false}\n'
