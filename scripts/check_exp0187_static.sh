#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${QBH_EXP0187_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0187/real_layer14_m1_hmxref}"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"

git -C "${project_root}" diff --check
grep -q 'weight.n = mxmem' "${project_root}/src/dsp/hmx_u8s8_projection.c"
grep -q 'QBH_WEIGHT_PACKED_W4_DIRECT_N' "${project_root}/include/probe_protocol.h"

"${python_exe}" - "${package}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["experiment"] == "EXP-0187"
assert manifest["reference_contract"]["valid_rows"] == 1
for name, record in manifest["files"].items():
    path = root / name
    assert path.stat().st_size == record["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
print(json.dumps({
    "experiment": "EXP-0187",
    "status": "pass",
    "package": str(root),
    "files_verified": len(manifest["files"]),
    "direct_n_symbol": True,
}))
PY
