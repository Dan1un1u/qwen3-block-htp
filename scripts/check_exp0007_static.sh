#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0006_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0007",
    max_compressed_slots=4,
    expanded_chunk_slots=8,
    candidate_compressed_slots=[2, 3, 4],
    candidate_chunk_tiles=[16, 32],
    candidate_hvx_workers=[6],
    dcvs_performance_vote=True,
    lock_free_hot_path_overlap_metrics=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
