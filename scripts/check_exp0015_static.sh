#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0014_static.sh")"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0015",
    resource_lifetime_candidates=[
        "transient_resources", "prepared_session"],
    warmup_runs=1,
    measured_runs=1,
    persistent_resources=["dcvs_vote", "hmx_power", "hmx_context",
                          "vtcm_2mib"],
    prepare_and_release_reported_separately=True,
    output_assembly="linked_2d_dma",
    projection_arithmetic_unchanged=True,
    runtime_resource_identity_gate_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
