#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh"
set -u

dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_probe_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"
android_readelf="${ANDROID_ROOT_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

for artifact in "${dsp_skel}" "${host_executable}" "${host_stub}"; do
    if [[ ! -f "${artifact}" ]]; then
        printf 'missing build artifact: %s\n' "${artifact}" >&2
        exit 1
    fi
done

"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${temporary_dir}/dsp.disassembly.txt"
"${hexagon_readelf}" -d "${dsp_skel}" \
    > "${temporary_dir}/dsp.dynamic.txt"
"${android_readelf}" -d "${host_executable}" \
    > "${temporary_dir}/host.dynamic.txt"
"${android_readelf}" -d "${host_stub}" \
    > "${temporary_dir}/stub.dynamic.txt"

grep -q 'bias = mxmem2' "${temporary_dir}/dsp.disassembly.txt"
grep -q 'mxclracc' "${temporary_dir}/dsp.disassembly.txt"
grep -q 'activation.ub = mxmem' "${temporary_dir}/dsp.disassembly.txt"
grep -q 'weight.b = mxmem' "${temporary_dir}/dsp.disassembly.txt"
grep -q ':after:cm:sat.ub = acc' "${temporary_dir}/dsp.disassembly.txt"
grep -q 'v0.cur = vmem' "${temporary_dir}/dsp.disassembly.txt"
grep -q 'vmem(.*) = v0' "${temporary_dir}/dsp.disassembly.txt"

if grep -Eiq 'Qnn|QAIRT' "${temporary_dir}"/*.dynamic.txt; then
    printf 'unexpected QNN/QAIRT runtime dependency\n' >&2
    exit 1
fi

printf '{"experiment":"EXP-0002","integer_hmx":true,'
printf '"u8_activation":true,"s8_weight":true,"u8_output":true,'
printf '"hvx_vtcm_copy":true,"qnn_dependency":false}\n'

if [[ -n "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    mkdir -p "${QBH_STATIC_OUTPUT_DIR}"
    cp "${temporary_dir}"/*.txt "${QBH_STATIC_OUTPUT_DIR}/"
fi
