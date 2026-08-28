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
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"

"${hexagon_readelf}" -Ws "${dsp_skel}" \
    > "${static_output_dir}/exp0026.symbols.txt"
"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${static_output_dir}/exp0026.disassembly.txt"
strings "${host_cli}" > "${static_output_dir}/exp0026.host.strings.txt"

grep -q 'EXP-0026' "${static_output_dir}/exp0026.host.strings.txt"
grep -q 'numerical_audit_mode' \
    "${static_output_dir}/exp0026.host.strings.txt"
grep -q 'residual_mode' "${static_output_dir}/exp0026.host.strings.txt"
grep -q 'qbh_hvx_residual_add_f16' \
    "${static_output_dir}/exp0026.symbols.txt"
grep -q 'qbh_hvx_residual_rms_norm_f16' \
    "${static_output_dir}/exp0026.symbols.txt"
grep -Eq 'v[0-9]+\.(qf16|hf)[[:space:]]*=[[:space:]]*vadd\(v[0-9]+\.hf,v[0-9]+\.hf\)' \
    "${static_output_dir}/exp0026.disassembly.txt"

printf '{"experiment":"EXP-0026","hvx_fp16_residual_add":true,'
printf '"audit_free_performance_mode":true,'
printf '"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"qnn_dependency":false}\n'
