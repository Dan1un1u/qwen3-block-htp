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
    experiment="EXP-0018",
    probe_abi=19,
    physical_plan="phased_group4_chunk64_dma_chain4",
    compressed_slots=8,
    expanded_slots=4,
    requested_hvx_workers=4,
    candidate_hvx_workers=[4],
    candidate_compressed_slots=[8],
    candidate_chunk_tiles=[64],
    selected_compressed_slots=8,
    selected_chunk_tiles=64,
    selected_hvx_workers=4,
    projection="gate_up_pair",
    projection_shapes={"paired_gate_up": [64, 2048, 12288]},
    hvx_group_barrier=True,
    linked_next_group_dma_prefetch=True,
    hvx_hmx_simultaneous_execution=False,
    qnn_dependency=False,
    down_two_chunk_candidate_disabled_after_diagnostic_stall=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
