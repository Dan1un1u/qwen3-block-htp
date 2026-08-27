#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0015_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0016",
    probe_abi=17,
    invocation_candidates=["two_call_control", "single_invocation"],
    projection_shapes={
        "two_call_control": [[64, 2048, 6144], [64, 2048, 6144]],
        "paired_gate_up": [64, 2048, 12288],
    },
    paired_output_channels=12288,
    paired_hmx_pairs_per_repeat=24576,
    paired_output_dma_descriptors=384,
    cache_maintenance="active_ranges_only",
    compared_total_weight_bytes_equal=True,
    compared_total_output_bytes_equal=True,
    prepared_session_required=True,
    qnn_dependency=False,
    runtime_canonical_output_equivalence_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
