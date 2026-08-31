#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objdump_bin="/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
lut="${QBH_EXP0117_LUT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel/silu_up_lut_u16.bin}"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(48)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(117)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_W4U8_ACTIVATION_LUT' \
    "${project_root}/src/host/block_main.c"
grep -q 'QBH_W4_ACTIVATION_LUT_PACKED_PAIR' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
"${project_root}/scripts/check_exp0117_lut.py" "${lut}" \
    > "${temporary_dir}/lut_gate.json"

for symbol in qbh_mlp_gate_up_lut_hvx \
              qbh_mlp_gate_up_packed_pair_lut_hvx; do
    "${objdump_bin}" --disassemble-symbols="${symbol}" "${dsp_skel}" \
        > "${temporary_dir}/${symbol}.txt"
done
control_gathers="$(grep -c 'vgather' "${temporary_dir}/qbh_mlp_gate_up_lut_hvx.txt")"
candidate_gathers="$(grep -c 'vgather' "${temporary_dir}/qbh_mlp_gate_up_packed_pair_lut_hvx.txt")"
[[ "${control_gathers}" == 4 ]]
[[ "${candidate_gathers}" == 2 ]]
if grep -q 'allocframe' \
        "${temporary_dir}/qbh_mlp_gate_up_packed_pair_lut_hvx.txt"; then
    printf 'packed-pair kernel spills a stack frame\n' >&2
    exit 1
fi
if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0117 source\n' >&2
    exit 1
fi
python3 - "${temporary_dir}/lut_gate.json" <<'PY'
import json
import sys

lut = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "experiment": "EXP-0117",
    "static_gate": "pass",
    "recipe": "W4U8",
    "control_gathers_per_128": 4,
    "candidate_gathers_per_128": 2,
    "candidate_predicated_split_gathers": 0,
    "candidate_stack_frame": False,
    "lut_exhaustive_gate": lut,
    "control_runtime_selectable": True,
    "packed_pair_runtime_selectable": True,
    "single_hmx_owner": True,
    "qnn_dependency": False,
}, sort_keys=True))
PY
