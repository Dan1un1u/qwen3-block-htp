#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
set -u

raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0019_static.sh")"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_nm="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-nm"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${temporary_dir}/dsp.disassembly.txt"
"${hexagon_nm}" -C "${dsp_skel}" > "${temporary_dir}/dsp.symbols.txt"
grep -q 'qbh_unpack_w4_to_s8_hvx_bitwise' \
    "${temporary_dir}/dsp.symbols.txt"

python3 - "${temporary_dir}/dsp.disassembly.txt" \
    "${temporary_dir}/bitwise_unpack.disassembly.txt" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
match = re.search(
    r"(?ms)^[^\n]*<qbh_unpack_w4_to_s8_hvx_bitwise>:\n"
    r"(.*?)(?=^[^\n]*<[^>]+>:\n|\Z)",
    source,
)
if match is None:
    raise SystemExit("bitwise unpack function missing from disassembly")
body = match.group(0)
if "vlut" in body:
    raise SystemExit("bitwise unpack unexpectedly contains a vector LUT")
if "vxor" not in body or "vsub" not in body or "vshuff" not in body:
    raise SystemExit("bitwise sign-extension sequence was not emitted")
pathlib.Path(sys.argv[2]).write_text(body)
PY

if [[ -n "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    mkdir -p "${QBH_STATIC_OUTPUT_DIR}"
    cp "${temporary_dir}/bitwise_unpack.disassembly.txt" \
        "${QBH_STATIC_OUTPUT_DIR}/"
fi

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0020",
    probe_abi=21,
    parent_gate_plan="stream32_gate_hvx2",
    bitwise_unpack_plan="stream32_gate_hvx2_bitwise",
    bitwise_signed_nibble_unpack=True,
    bitwise_unpack_vlut=False,
    exactly_two_hvx_producers=True,
    persistent_ordered_hmx_consumer=True,
    whole_group_barrier=False,
    qnn_dependency=False,
)
print(json.dumps(gate, separators=(",", ":")))
PY
