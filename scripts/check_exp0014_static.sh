#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0013_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0014",
    output_assembly_candidates=["scalar_memcpy", "linked_2d_dma"],
    output_dma_descriptor_type="linked_2d",
    output_dma_descriptor_counts={"gate_up": 192, "down": 64},
    output_dma_source="VTCM HMX tile-major output",
    output_dma_destination="shared DDR row-major output",
    weight_dma_hvx_hmx_pipeline_unchanged=True,
    integer_hmx_work_unchanged=True,
    vtcm_request_bytes=2097152,
    runtime_exactness_and_dma_counter_gate_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
