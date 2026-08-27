#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_gate="$(QBH_STATIC_OUTPUT_DIR="${QBH_STATIC_OUTPUT_DIR:-}" \
    "${project_root}/scripts/check_exp0011_static.sh")"

required_symbols=(
    qbh_start_linked_weight_bundles
    qbh_wait_linked_descriptor
    QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS
    dma_descriptor_completion_count
    dma_descriptor_timeout_count
)
search_symbol() {
    local symbol="$1"
    if [[ -x /usr/bin/rg ]]; then
        /usr/bin/rg -q "${symbol}" "${project_root}/src" "${project_root}/include"
    else
        grep -R -q -- "${symbol}" "${project_root}/src" "${project_root}/include"
    fi
}
for symbol in "${required_symbols[@]}"; do
    if ! search_symbol "${symbol}"; then
        printf 'missing linked-DMA implementation symbol: %s\n' "${symbol}" >&2
        exit 1
    fi
done

RAW_GATE="${raw_gate}" python3 - <<'PY'
import json
import os

gate = json.loads(os.environ["RAW_GATE"])
gate.update(
    experiment="EXP-0012",
    projections=["gate_up", "down"],
    expanded_s8_dma_plans=["batch1", "contiguous_batch2", "linked_chain2"],
    packed_w4_dma_plans=[
        "batch1", "contiguous_batch2", "linked_chain2", "linked_chain4"
    ],
    descriptor_completion_publication=True,
    bounded_descriptor_timeout=True,
    integer_hmx_work_unchanged=True,
    logical_weight_bundles_unchanged=True,
    runtime_descriptor_count_gate_required=True,
)
print(json.dumps(gate, separators=(",", ":")))
PY
