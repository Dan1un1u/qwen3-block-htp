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

grep -q 'EXP-0030' "${static_output_dir}/host.strings.txt"
grep -q 'multi_worker_silu' "${static_output_dir}/host.strings.txt"
grep -q 'streaming' "${static_output_dir}/host.strings.txt"
grep -q 'qbh_hvx_silu_multiply_f16_vectors' \
    "${static_output_dir}/dsp.symbols.txt"
grep -q 'qbh_hvx_silu_multiply_f16_channel64' \
    "${static_output_dir}/dsp.symbols.txt"
grep -q 'activation.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'weight.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"

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
hvx = (root / "src/dsp/hvx_fp16_ops.c").read_text()
idl = (root / "include/qwen3_probe.idl").read_text()

required = (
    "QBH_BLOCK_EXPERIMENT UINT32_C(30)",
    "QBH_BLOCK_MLP_MULTI_WORKER_SILU",
    "QBH_BLOCK_MLP_STREAMING",
    "qbh_hvx_pool_silu(",
    "qbh_silu_pool_run_chunks(",
    "qbh_run_w4f16_interleaved_gate_up(",
    "qbh_mlp_stream_publish_up_group(",
    "qbh_pack_fp16_activation_channel64(",
    "qbh_hvx_silu_multiply_f16_vectors(",
    "mlp_silu_worker_work_ticks",
    "mlp_silu_pool_wait_ticks",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
)
for fragment in required:
    if fragment not in implementation and fragment not in protocol and fragment not in hvx:
        raise SystemExit(f"missing EXP-0030 contract fragment: {fragment}")

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

printf '{"experiment":"EXP-0030","multi_worker_silu":true,'
printf '"interleaved_streaming_mlp":true,'
printf '"fixed_vtcm_request_bytes":8388608,'
printf '"intermediate_ddr_allowed":false,"qnn_dependency":false}\n'
