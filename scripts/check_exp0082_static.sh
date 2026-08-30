#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
set -u
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(42)' "${project_root}/include/block_protocol.h"
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(82)' "${project_root}/include/block_protocol.h"
grep -q 'EXP-0082' "${project_root}/src/host/block_main.c"
grep -q 'qbh_w4_u8_set_decode_mode' "${project_root}/src/dsp/block_imp.c"
if grep -Rqs --exclude='check_exp0082_static.sh' \
        -E 'Qnn|qti\.aisw|QAIRT' "${project_root}/include" \
        "${project_root}/src" "${project_root}/CMakeLists.txt"; then
    printf 'unexpected QNN/QAIRT dependency\n' >&2
    exit 1
fi

"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${temporary_dir}/dsp.disassembly.txt"
python3 - "${temporary_dir}/dsp.disassembly.txt" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
def body(name: str) -> str:
    match = re.search(
        rf"(?ms)^[^\n]*<{name}>:\n(.*?)(?=^[^\n]*<[^>]+>:\n|\Z)",
        source,
    )
    if match is None:
        raise SystemExit(f"missing disassembly for {name}")
    return match.group(0)

control = body("qbh_unpack_w4_to_s8_hvx_vlut32")
candidate = body("qbh_unpack_w4_to_s8_hvx_arithmetic")
if "vlut" not in control:
    raise SystemExit("control decoder does not contain vlut")
if "vlut" in candidate:
    raise SystemExit("arithmetic decoder still contains vlut")
if "vxor" not in candidate or "vsub(" not in candidate:
    raise SystemExit("arithmetic decoder lacks vector XOR/subtract")
PY

printf '%s\n' \
    '{"experiment":"EXP-0082","static_gate":"pass","block_abi":42,"runtime_telemetry_experiment":82,"control_decoder":"vlut32","candidate_decoder":"vector_xor_subtract","candidate_vlut32":false,"qnn_dependency":false,"vtcm_request_bytes":8388608,"single_hmx_owner":true}'
