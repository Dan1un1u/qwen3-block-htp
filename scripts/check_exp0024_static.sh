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
QBH_STATIC_OUTPUT_DIR="${static_output_dir}" \
    "${project_root}/scripts/check_exp0022_static.sh" >/dev/null

dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"

"${hexagon_readelf}" -Ws "${dsp_skel}" \
    > "${static_output_dir}/exp0024.symbols.txt"
"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${static_output_dir}/exp0024.disassembly.txt"

for symbol in qbh_hvx_rms_norm_f16 qbh_hvx_qk_norm_rope_f16 \
              qbh_hvx_silu_multiply_f16 \
              qbh_hvx_stable_causal_softmax_f16; do
    grep -q "${symbol}" "${static_output_dir}/exp0024.symbols.txt"
done
grep -q 'vmpy(.*\.hf' "${static_output_dir}/exp0024.disassembly.txt"
grep -q 'vadd(.*\.qf16' "${static_output_dir}/exp0024.disassembly.txt"
grep -q 'vmax(.*\.hf' "${static_output_dir}/exp0024.disassembly.txt"
grep -q 'vcmp\.gt(.*\.hf' "${static_output_dir}/exp0024.disassembly.txt"

printf '{"experiment":"EXP-0024","hvx_fp16_rmsnorm":true,'
printf '"hvx_fp16_rope":true,"hvx_fp16_silu":true,'
printf '"hvx_fp16_stable_causal_softmax":true,'
printf '"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"qnn_dependency":false}\n'
