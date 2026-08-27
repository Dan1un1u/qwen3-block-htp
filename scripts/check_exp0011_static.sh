#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0010_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0011",
    projections=["gate_up", "down"],
    expanded_s8_dma_batches=[1, 2],
    packed_w4_dma_batches=[1, 2, 4],
    gate_up_chunk_tiles=64,
    down_chunk_tiles=96,
    integer_hmx_work_unchanged=True,
    logical_weight_bundles_unchanged=True,
    runtime_dma_count_gate_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
