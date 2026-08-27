#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0009_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0010",
    projection="gate_up",
    inherited_chunk_tiles=32,
    candidate_chunk_tiles=[64],
    internal_hmx_stream_tiles=32,
    control_publications_per_output=2,
    candidate_publications_per_output=[1],
    expanded_chunk_slots=8,
    compressed_slots=4,
    requested_hvx_workers=6,
)
print(json.dumps(gate, separators=(",", ":")))
PY
