#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0012_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0013",
    projections=["gate_up", "down"],
    vtcm_request_bytes=2097152,
    max_compressed_slots=8,
    expanded_chunk_slots_max=8,
    candidate_compressed_slots=[8],
    compressed_slots=[4, 8],
    selected_compressed_slots=None,
    parent_ring={"compressed": 4, "expanded": 8, "dma_batch": 2},
    isolation_ring={"compressed": 4, "expanded": 7, "dma_batch": 2},
    candidate_ring={"compressed": 8, "expanded": 7, "dma_batch": 4},
    candidate_publication=["contiguous", "linked_descriptor"],
    integer_hmx_work_unchanged=True,
    logical_weight_bundles_unchanged=True,
    runtime_ring_and_vtcm_gate_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
