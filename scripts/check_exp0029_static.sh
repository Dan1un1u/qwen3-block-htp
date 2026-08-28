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

dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
host_cli="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"
android_readelf="${ANDROID_ROOT_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"

mkdir -p "${static_output_dir}"
for artifact in "${dsp_skel}" "${host_cli}" "${host_stub}"; do
    [[ -f "${artifact}" ]] || {
        printf 'missing build artifact: %s\n' "${artifact}" >&2
        exit 1
    }
done

"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${static_output_dir}/dsp.disassembly.txt"
"${hexagon_readelf}" -Ws "${dsp_skel}" \
    > "${static_output_dir}/dsp.symbols.txt"
"${hexagon_readelf}" -d "${dsp_skel}" \
    > "${static_output_dir}/dsp.dynamic.txt"
"${android_readelf}" -d "${host_cli}" \
    > "${static_output_dir}/host.dynamic.txt"
"${android_readelf}" -d "${host_stub}" \
    > "${static_output_dir}/stub.dynamic.txt"
strings "${host_cli}" > "${static_output_dir}/host.strings.txt"

grep -q 'EXP-0029' "${static_output_dir}/host.strings.txt"
grep -q 'combined_hvx' "${static_output_dir}/host.strings.txt"
grep -q 'vscatter' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'vshuff' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'activation.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'weight.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'cvt.hf = acc' "${static_output_dir}/dsp.disassembly.txt"

if grep -Eiq 'Qnn|QAIRT' "${static_output_dir}"/*.dynamic.txt; then
    printf 'unexpected QNN/QAIRT runtime dependency\n' >&2
    exit 1
fi

PROJECT_ROOT="${project_root}" python3 - <<'PY'
import os
import pathlib
import re

root = pathlib.Path(os.environ["PROJECT_ROOT"])
implementation = (root / "src/dsp/block_imp.c").read_text()
protocol = (root / "include/block_protocol.h").read_text()
idl = (root / "include/qwen3_probe.idl").read_text()

required = (
    "QBH_BLOCK_EXPERIMENT UINT32_C(29)",
    "QBH_BLOCK_ATTENTION_PACK_QK_HVX",
    "QBH_BLOCK_ATTENTION_PACK_AV_HVX",
    "qbh_pack_fp16_weight_rows_hvx(",
    "qbh_pack_fp16_weight_transposed_hvx(",
    "Q6_vscatter_RMVwV(",
    "Q6_W_vshuff_VVR(",
    "packed_qk_kv_head",
    "packed_av_kv_head",
    "attention_qk_pack_ticks",
    "attention_av_pack_ticks",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
)
for fragment in required:
    if fragment not in implementation and fragment not in protocol:
        raise SystemExit(f"missing EXP-0029 contract fragment: {fragment}")

header_offsets = set(re.findall(
    r"shared\s*\+\s*header->([a-z_]+_offset)", implementation))
projection_offsets = set(re.findall(
    r"shared\s*\+\s*desc->([a-z_]+_offset)", implementation))
if header_offsets != {"input_offset", "output_offset"}:
    raise SystemExit(f"illegal header DDR boundary: {sorted(header_offsets)}")
if projection_offsets != {"weight_offset", "scale_offset", "bias_offset"}:
    raise SystemExit(f"illegal projection DDR boundary: {sorted(projection_offsets)}")
if idl.count("AEEResult run_block(") != 1:
    raise SystemExit("run_block must remain exactly one FastRPC method")
if re.search(r"intermediate_[a-z_]+_offset", protocol):
    raise SystemExit("DSP ABI exposes an intermediate DDR tensor")
PY

printf '{"experiment":"EXP-0029","fp16_hmx":true,'
printf '"hvx_qk_vscatter":true,"hvx_av_vshuff":true,'
printf '"gqa_kv_pack_reuse":true,"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"qnn_dependency":false}\n'
