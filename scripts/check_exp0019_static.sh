#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0017_static.sh")"

grep -q 'qbh_hmx_accumulate_u8s8_streaming' \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"
grep -q 'stream_ready_generation' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"
grep -q 'QBH_STREAM_READY_TIMEOUT_TICKS' \
    "${project_root}/src/dsp/w4_parallel_pipeline.c"

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0019",
    probe_abi=20,
    physical_contract="streaming_microchunk_handoff",
    stream_region_tiles=32,
    parent_publication_tiles={"paired_gate_up": 64, "down": 96},
    candidate_hvx_workers=[2, 3, 4, 6],
    generation_tagged_publication=True,
    persistent_ordered_hmx_consumer=True,
    whole_group_barrier=False,
    bounded_ready_timeout=True,
    qnn_dependency=False,
)
print(json.dumps(gate, separators=(",", ":")))
PY
