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

grep -q 'EXP-0035' "${output_dir}/host.strings.txt"
grep -q 'adaptive_down96_gate4_dma8_cross_q_prefetch' \
    "${output_dir}/host.strings.txt"
grep -q 'q_inbound_prefetch_overlap_window_ticks' \
    "${output_dir}/host.strings.txt"
grep -q 'qbh_unpack_w4_to_f16_hvx' "${output_dir}/dsp.symbols.txt"
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
    "QBH_BLOCK_EXPERIMENT UINT32_C(35)",
    "QBH_BLOCK_ABI_VERSION UINT32_C(18)",
    "QBH_BLOCK_W4F16_PIPELINE_ADAPTIVE_DOWN96_GATE4_DMA8_CROSS_Q_PREFETCH",
    "qbh_w4f16_start_q_inbound_prefetch(",
    "q_inbound_prefetch_start_count",
    "q_inbound_prefetch_completion_count",
    "q_inbound_prefetch_consume_count",
    "q_inbound_prefetch_bytes",
    "q_inbound_prefetch_descriptor_count",
    "q_inbound_prefetch_lifetime_ticks",
    "q_inbound_prefetch_wait_ticks",
    "q_inbound_prefetch_overlap_window_ticks",
    "qbh_run_attributed_qkv_projection(",
)
for fragment in required:
    if fragment not in implementation and fragment not in protocol and fragment not in host:
        raise SystemExit(f"missing EXP-0035 contract fragment: {fragment}")

run_one_block = implementation.index("static int qbh_run_one_block(")
input_dma = implementation.index(
    "shared + header->input_offset", run_one_block)
q_prefetch = implementation.index(
    "qbh_w4f16_start_q_inbound_prefetch(", input_dma)
input_norm = implementation.index(
    "if (header->variant == QBH_BLOCK_W4U8)", q_prefetch)
q_projection = implementation.index(
    "header, shared, QBH_BLOCK_PROJ_Q", input_norm)
if not input_dma < q_prefetch < input_norm < q_projection:
    raise SystemExit("Q inbound prefetch is outside the approved schedule boundary")

qkv_calls = re.findall(
    r"qbh_run_attributed_qkv_projection\(\s*header,\s*shared,\s*"
    r"(QBH_BLOCK_PROJ_[QKV])", implementation)
if qkv_calls != ["QBH_BLOCK_PROJ_Q", "QBH_BLOCK_PROJ_K", "QBH_BLOCK_PROJ_V"]:
    raise SystemExit(f"unexpected attributed QKV call order: {qkv_calls}")

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

printf '{"experiment":"EXP-0035","q_inbound_prefetch":true,'
printf '"moved_transfer_only":true,"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"single_hmx_owner":true,'
printf '"qnn_dependency":false}\n'
