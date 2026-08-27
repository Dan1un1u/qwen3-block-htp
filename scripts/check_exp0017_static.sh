#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0016_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0017",
    probe_abi=18,
    scale_placements=["hvx_prescale", "hmx_postscale"],
    projection_shapes={
        "paired_gate_up": [64, 2048, 12288],
        "down": [64, 6144, 2048],
    },
    same_integer_hmx_family=True,
    same_logical_s8_carrier=True,
    prepared_session_required=True,
    qnn_dependency=False,
    runtime_exact_equivalence_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
