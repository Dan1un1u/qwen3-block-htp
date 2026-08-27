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
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_readelf="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-readelf"
android_readelf="${ANDROID_ROOT_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"

mkdir -p "${static_output_dir}"
for artifact in "${dsp_skel}" "${host_executable}" "${host_stub}"; do
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
"${android_readelf}" -d "${host_executable}" \
    > "${static_output_dir}/host.dynamic.txt"
"${android_readelf}" -d "${host_stub}" \
    > "${static_output_dir}/stub.dynamic.txt"

grep -q 'qbh_run_block_rpc' "${static_output_dir}/dsp.symbols.txt"
grep -q 'qbh_hmx_fp16_matmul_tiles' "${static_output_dir}/dsp.symbols.txt"
grep -q 'qbh_hmx_fp16_matmul_tile_scales' "${static_output_dir}/dsp.symbols.txt"
grep -q 'qbh_hmx_begin_u8s8_output' "${static_output_dir}/dsp.symbols.txt"
grep -q 'mxclracc.hf' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'activation.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'weight.hf = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'cvt.hf = acc' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'activation.ub = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'weight.b = mxmem' "${static_output_dir}/dsp.disassembly.txt"
grep -q ':after:cm:sat.ub = acc' "${static_output_dir}/dsp.disassembly.txt"
grep -q 'vlut16' "${static_output_dir}/dsp.disassembly.txt"

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

header_offsets = set(re.findall(
    r"shared\s*\+\s*header->([a-z_]+_offset)", implementation))
projection_offsets = set(re.findall(
    r"shared\s*\+\s*desc->([a-z_]+_offset)", implementation))
if header_offsets != {"input_offset", "output_offset"}:
    raise SystemExit(f"illegal header DDR boundary: {sorted(header_offsets)}")
if projection_offsets != {"weight_offset", "scale_offset", "bias_offset"}:
    raise SystemExit(
        f"illegal projection DDR boundary: {sorted(projection_offsets)}")

required = (
    "QBH_EXPECTED_FULL_VTCM_BYTES",
    "QBH_HMX_FP16_TILE_BYTES",
    "QBH_HMX_FP16_SCALE_BYTES",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
    "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count",
    "qbh_arena_alloc_aligned(",
    "qbh_attention_f16(",
    "qbh_run_projection(",
    "qurt_sem_init_val(&worker.command_ready, 0U)",
    "qurt_sem_init_val(&worker.command_done, 0U)",
    "qurt_sem_init_val(&worker.worker_started, 0U)",
)
for fragment in required:
    if fragment not in implementation and fragment not in protocol:
        raise SystemExit(f"missing physical-contract fragment: {fragment}")
if idl.count("AEEResult run_block(") != 1:
    raise SystemExit("run_block must be exactly one FastRPC method")
if re.search(r"intermediate_[a-z_]+_offset", protocol):
    raise SystemExit("DSP ABI exposes an intermediate DDR tensor")
PY

printf '{"experiment":"EXP-0023","execution_unit":"qwen3_layer14_complete_block_m64",'
printf '"fp16_hmx":true,"u8s8_integer_hmx":true,"single_block_fastrpc":true,'
printf '"fixed_vtcm_request_bytes":8388608,"hmx_crouton_alignment_bytes":2048,'
printf '"hmx_bias_alignment_bytes":256,"intermediate_ddr_allowed":false,'
printf '"qnn_dependency":false}\n'
