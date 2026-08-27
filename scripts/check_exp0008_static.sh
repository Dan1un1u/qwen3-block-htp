#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh" >/dev/null
set -u

raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0007_static.sh")"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
hexagon_objdump="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-llvm-objdump"
hexagon_nm="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-nm"
temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

"${hexagon_objdump}" -d --no-show-raw-insn "${dsp_skel}" \
    > "${temporary_dir}/dsp.disassembly.txt"
"${hexagon_nm}" -C "${dsp_skel}" \
    > "${temporary_dir}/dsp.symbols.txt"
grep -q 'qbh_unpack_w4_to_s8_hvx' "${temporary_dir}/dsp.symbols.txt"

python3 - "${temporary_dir}/dsp.disassembly.txt" \
    "${temporary_dir}/postscale_unpack.disassembly.txt" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
match = re.search(
    r"(?ms)^[^\n]*<qbh_unpack_w4_to_s8_hvx>:\n"
    r"(.*?)(?=^[^\n]*<[^>]+>:\n|\Z)",
    source,
)
if match is None:
    raise SystemExit("postscale unpack function missing from disassembly")
body = match.group(0)
if "vlut" not in body:
    raise SystemExit("postscale unpack does not contain an HVX LUT")
if re.search(r"vmpy|vpack.*sat", body):
    raise SystemExit("postscale unpack still contains per-weight scale/pack work")
pathlib.Path(sys.argv[2]).write_text(body)
PY

if [[ -n "${QBH_STATIC_OUTPUT_DIR:-}" ]]; then
    mkdir -p "${QBH_STATIC_OUTPUT_DIR}"
    cp "${temporary_dir}/postscale_unpack.disassembly.txt" \
        "${QBH_STATIC_OUTPUT_DIR}/"
fi

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0008",
    hmx_post_accumulator_scale=True,
    postscale_unpack_hvx_lut=True,
    postscale_unpack_per_weight_multiply=False,
    integer_scale_range=[1, 18],
    selected_compressed_slots=4,
    selected_chunk_tiles=32,
    selected_hvx_workers=6,
)
print(json.dumps(gate, separators=(",", ":")))
PY
