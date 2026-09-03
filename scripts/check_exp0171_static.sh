#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp)"
trap 'rm -f "${temporary}"' EXIT

cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(74)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(171)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DECODE_SWIGLU_ROW4' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_SWIGLU' src/host/block_main.c
grep -q 'activation_elements' src/dsp/w4_parallel_pipeline.c
grep -q 'QBH_MLP_HVX_VECTOR_BYTES' src/dsp/block_imp.c
test -f "${skel}"
"${objdump}" -d --no-show-raw-insn --disassemble-symbols=qbh_mlp_gate_up_lut_hvx "${skel}" > "${temporary}"
grep -q 'vgather' "${temporary}"
grep -q 'vpack' "${temporary}"
grep -q 'loop0' "${temporary}"
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0171.sh scripts/deploy_exp0171.sh \
    scripts/run_exp0171.sh scripts/run_exp0171_formal.sh \
    scripts/check_exp0171_static.sh
python3 -m py_compile scripts/summarize_exp0171.py
printf '%s\n' '{"experiment":"EXP-0171","static_gate":"pass","variant":"W4U8","source_parent":"EXP-0170","control":"full_tile","candidate":"decode_row4","qnn_dependency":false}'
