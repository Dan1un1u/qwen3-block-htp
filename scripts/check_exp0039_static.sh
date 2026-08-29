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

grep -q 'EXP-0039' "${output_dir}/host.strings.txt"
grep -q 'vtcm_plan_mode' "${output_dir}/host.strings.txt"
grep -q 'gate_up_deep' "${output_dir}/host.strings.txt"
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
protocol = (root / "include/block_protocol.h").read_text()
host = (root / "src/host/block_main.c").read_text()
idl = (root / "include/qwen3_probe.idl").read_text()

required = (
    "QBH_BLOCK_EXPERIMENT UINT32_C(39)",
    "QBH_BLOCK_ABI_VERSION UINT32_C(19)",
    "QBH_BLOCK_VTCM_PLAN_PHASE_OVERLAY",
    "QBH_BLOCK_VTCM_PLAN_GATE_UP_DEEP",
    "QBH_BLOCK_MLP_CROUTON_DEEP_RING_SLOTS",
    "expanded_weight_lookahead",
    "vtcm_phase_overlay_bytes",
    "vtcm_ring_compaction_bytes",
    "vtcm_lookahead_bytes",
    "w4f16_gate_up_lookahead_hmx_count",
)
combined = implementation + protocol + host
for fragment in required:
    if fragment not in combined:
        raise SystemExit(f"missing EXP-0039 contract fragment: {fragment}")

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

printf '{"experiment":"EXP-0039","stage":"static","abi":19,'
printf '"phase_overlay":true,"gate_up_deep":true,'
printf '"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"single_hmx_owner":true,'
printf '"qnn_dependency":false}\n'
