#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
static_output_dir="${QBH_STATIC_OUTPUT_DIR:-$(mktemp -d)}"
owns_output_dir=0
if [[ -z "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    owns_output_dir=1
fi
trap 'if [[ "${owns_output_dir}" == "1" ]]; then rm -rf -- "${static_output_dir}"; fi' EXIT

mkdir -p "${static_output_dir}"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${static_output_dir}" \
    "${project_root}/scripts/check_exp0019_static.sh")"

dsp_symbols="${static_output_dir}/dsp.symbols.txt"
dsp_disassembly="${static_output_dir}/dsp.disassembly.txt"
host_executable="${project_root}/android_ReleaseG_aarch64/ship/qwen3_mlp_cli"

[[ -f "${host_executable}" ]] || {
    printf 'missing MLP host executable: %s\n' "${host_executable}" >&2
    exit 1
}

grep -q 'qbh_run_mlp_rpc' "${dsp_symbols}"
grep -q 'qbh_mlp_gate_up_hvx' "${dsp_symbols}"
grep -q 'qbh_run_chunked_w4_pipeline_mlp' "${dsp_symbols}"
grep -q 'qbh_chunked_hmx_main' "${dsp_symbols}"
grep -q 'mxclracc' "${dsp_disassembly}"
grep -q 'activation.ub = mxmem' "${dsp_disassembly}"
grep -q 'weight.b = mxmem' "${dsp_disassembly}"
grep -Eq 'vmpy.*\.b|vmpy.*b' "${dsp_disassembly}"
grep -Eq 'vpack.*sat' "${dsp_disassembly}"

PROJECT_ROOT="${project_root}" python3 - <<'PY'
import pathlib
import re
import os

root = pathlib.Path(os.environ["PROJECT_ROOT"])
implementation = (root / "src/dsp/mlp_imp.c").read_text()
pipeline = (root / "src/dsp/w4_parallel_pipeline.c").read_text()
protocol = (root / "include/mlp_protocol.h").read_text()
idl = (root / "include/qwen3_probe.idl").read_text()

shared_offsets = set(re.findall(r"shared\s*\+\s*header->([a-z_]+_offset)", implementation))
legal_offsets = {
    "input_offset",
    "gate_up_weight_offset",
    "down_weight_offset",
    "output_offset",
}
if shared_offsets != legal_offsets:
    raise SystemExit(
        f"illegal or missing DSP shared-memory boundary: {sorted(shared_offsets)}")

required_fragments = (
    "QBH_MLP_GATE_UP_PAIR_SLOTS UINT32_C(8)",
    "header->intermediate_ddr_read_bytes = 0U",
    "header->intermediate_ddr_write_bytes = 0U",
    "header->intermediate_dma_descriptor_count = 0U",
    "header->intermediate_spill_fill_count = 0U",
    "header->gate_up_output_dma_descriptor_count = 0U",
    "header->middle_dma_descriptor_count = 0U",
    "header->gate_up_full_tensor_materialized = 0U",
    "shared + header->down_weight_offset, vtcm, vtcm",
    "shared + header->output_offset",
)
for fragment in required_fragments:
    if fragment not in implementation:
        raise SystemExit(f"missing physical-contract fragment: {fragment}")

if "QBH_MLP_INTERMEDIATE_BYTES" not in protocol:
    raise SystemExit("missing fixed VTCM middle-tensor size")
if idl.count("AEEResult run_mlp(") != 1:
    raise SystemExit("run_mlp must be one FastRPC method")
if "qbh_mlp_gate_up_hvx(" not in pipeline:
    raise SystemExit("Gate/Up tiles are not handed directly to the HVX activation")
if "rpcmem" in implementation or re.search(r"intermediate_[a-z_]+_offset", protocol):
    raise SystemExit("DSP ABI exposes an illegal intermediate DDR buffer")
PY

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0021",
    mlp_abi=2,
    execution_unit="qwen3_middle_block_mlp",
    physical_contract="paired_gate_up_tile_to_vtcm_middle_to_down",
    legal_ddr_buffers=["input", "packed_gate_up_weights", "packed_down_weights", "final_output"],
    illegal_ddr_buffers=["gate", "up", "activated_product", "down_input"],
    gate_up_pair_slots=8,
    gate_up_pair_ring_bytes=32768,
    middle_vtcm_bytes=393216,
    final_output_vtcm_bytes=131072,
    vtcm_request_bytes=2097152,
    one_prepared_measured_fastrpc=True,
    warmup_fastrpc_calls=1,
    measured_fastrpc_calls=1,
    direct_gate_up_hvx_handoff=True,
    direct_vtcm_middle_to_down=True,
    full_gate_up_tensor_materialized=False,
    intermediate_ddr_allowed=False,
    intermediate_dma_allowed=False,
    spill_fill_allowed=False,
    final_output_dma_only=True,
    qnn_dependency=False,
)
print(json.dumps(gate, separators=(",", ":")))
PY
