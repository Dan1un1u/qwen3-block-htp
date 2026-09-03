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
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(172)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_DECODE_GATE_UP_HVX_GEMV' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_GATE_UP_COMPUTE' src/host/block_main.c
grep -q 'qbh_w4u8_gate_up_hvx_gemv_batch' src/dsp/block_imp.c
test -f "${skel}"
"${objdump}" -d --no-show-raw-insn "${skel}" > "${temporary}"
grep -q 'vrmpy' "${temporary}"
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0172.sh scripts/deploy_exp0172.sh \
    scripts/run_exp0172.sh scripts/check_exp0172_static.sh

printf '%s\n' '{"experiment":"EXP-0172","static_gate":"pass","variant":"W4U8","source_parent":"EXP-0170","control":"hmx","candidate":"hvx_gemv","qnn_dependency":false}'
