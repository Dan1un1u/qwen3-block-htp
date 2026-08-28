#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
set -u

output_dir="${QBH_STATIC_OUTPUT_DIR:-$(mktemp -d)}"
owns_output=0
if [[ -z "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    owns_output=1
fi
trap 'if [[ "${owns_output}" == "1" ]]; then rm -rf -- "${output_dir}"; fi' EXIT

dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
host_cli="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"
android_readelf="${ANDROID_ROOT_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"

mkdir -p "${output_dir}"
"${hexagon_readelf}" -Ws "${dsp_skel}" > "${output_dir}/dsp.symbols.txt"
"${hexagon_readelf}" -d "${dsp_skel}" > "${output_dir}/dsp.dynamic.txt"
"${android_readelf}" -d "${host_cli}" > "${output_dir}/host.dynamic.txt"
"${android_readelf}" -d "${host_stub}" > "${output_dir}/stub.dynamic.txt"
strings "${host_cli}" > "${output_dir}/host.strings.txt"

grep -q 'EXP-0038' "${output_dir}/host.strings.txt"
grep -q 'crouton_boundary_mode' "${output_dir}/host.strings.txt"
grep -q 'qbh_hvx_qk_norm_rope_f16_crouton_head' \
    "${output_dir}/dsp.symbols.txt"
grep -q 'qbh_hvx_rms_norm_f16_crouton' "${output_dir}/dsp.symbols.txt"
grep -q 'qbh_hvx_residual_rms_norm_f16_crouton' \
    "${output_dir}/dsp.symbols.txt"
grep -q 'qbh_hmx_fp16_matmul_tile_scales' "${output_dir}/dsp.symbols.txt"
if grep -Eiq 'Qnn|QAIRT' "${output_dir}"/*.dynamic.txt; then
    printf 'unexpected QNN/QAIRT runtime dependency\n' >&2
    exit 1
fi

PROJECT_ROOT="${project_root}" python3 - <<'PY'
import os
import pathlib
import re

root = pathlib.Path(os.environ["PROJECT_ROOT"])
implementation = (root / "src/dsp/block_imp.c").read_text()
hvx = (root / "src/dsp/hvx_fp16_ops.c").read_text()
protocol = (root / "include/block_protocol.h").read_text()
host = (root / "src/host/block_main.c").read_text()
idl = (root / "include/qwen3_probe.idl").read_text()

required = (
    "QBH_BLOCK_EXPERIMENT UINT32_C(38)",
    "QBH_BLOCK_ABI_VERSION UINT32_C(18)",
    "QBH_BLOCK_CROUTON_BOUNDARY_QKV",
    "QBH_BLOCK_CROUTON_BOUNDARY_AV_TO_O",
    "QBH_BLOCK_CROUTON_BOUNDARY_INPUT_NORM",
    "QBH_BLOCK_CROUTON_BOUNDARY_POST_NORM",
    "qbh_hvx_qk_norm_rope_f16_crouton_head(",
    "qbh_hvx_rms_norm_f16_crouton(",
    "qbh_hvx_residual_rms_norm_f16_crouton(",
    "crouton_q_operand_mismatch_count",
    "crouton_k_operand_mismatch_count",
    "crouton_v_operand_mismatch_count",
    "crouton_qkv_transform_ticks",
    "crouton_av_o_copy_ticks",
    "crouton_norm_store_ticks",
)
combined = implementation + hvx + protocol + host
for fragment in required:
    if fragment not in combined:
        raise SystemExit(f"missing EXP-0038 contract fragment: {fragment}")

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

printf '{"experiment":"EXP-0038","stage":"static","abi":18,'
printf '"qkv_crouton":true,"av_to_o_crouton":true,'
printf '"input_norm_crouton":true,"post_norm_crouton":true,'
printf '"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"single_hmx_owner":true,'
printf '"qnn_dependency":false}\n'
