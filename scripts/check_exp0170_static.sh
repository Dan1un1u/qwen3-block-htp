#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
temporary="$(mktemp)"
trap 'rm -f "${temporary}"' EXIT

cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(73)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(170)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DECODE_SOFTMAX_HVX_TILE4' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_SOFTMAX' src/host/block_main.c
grep -q 'qbh_attention_u8_requant_softmax_dynamic_hvx_tile4' \
    src/dsp/attention_u8_core.c
grep -q 'dynamic_hvx_tile4_call_count' include/attention_u8_core.h
test -f "${skel}"
"${objdump}" -d --no-show-raw-insn \
    --disassemble-symbols=qbh_attention_u8_requant_softmax_dynamic \
    "${skel}" > "${temporary}"
grep -q 'vlut32' "${temporary}"
grep -q 'vmax' "${temporary}"
grep -q 'vshuff' "${temporary}"
grep -q 'vmpy' "${temporary}"
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0170.sh scripts/deploy_exp0170.sh \
    scripts/run_exp0170.sh scripts/run_exp0170_formal.sh \
    scripts/check_exp0170_static.sh
python3 -m py_compile scripts/summarize_exp0170.py

printf '%s\n' '{"experiment":"EXP-0170","static_gate":"pass","variant":"W4U8","source_parent":"EXP-0169","control":"scalar","candidate":"hvx_tile4","qnn_dependency":false}'
