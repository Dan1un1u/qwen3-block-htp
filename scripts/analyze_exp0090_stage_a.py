#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


def load_jsonl(path: Path) -> dict:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if line.startswith("{"):
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no JSON record in {path}")
    return records[-1]


def med(values):
    return statistics.median(values)


root = Path(sys.argv[1])
ordinary = load_jsonl(root / "ordinary_control.jsonl")
correctness = load_jsonl(root / "correctness_audit.jsonl")
audits = [load_jsonl(path) for path in sorted((root / "audit_runs").glob("*.jsonl"))]
if len(audits) != 7:
    raise RuntimeError(f"expected 7 audit runs, found {len(audits)}")

claim_counts = [int(x["w4u8_mlp_activation_queue_claim_count"]) for x in audits]
wait_sums = [int(x["w4u8_mlp_activation_queue_wait_ticks"]) for x in audits]
wait_maxima = [int(x["w4u8_mlp_activation_queue_wait_max_ticks"]) for x in audits]
ahead_sums = [int(x["w4u8_mlp_activation_queue_tasks_ahead_sum"]) for x in audits]
ahead_maxima = [int(x["w4u8_mlp_activation_queue_tasks_ahead_max"]) for x in audits]
depth_maxima = [int(x["w4u8_mlp_activation_queue_depth_max"]) for x in audits]
activation_work = [int(x["w4u8_mlp_activation_work_ticks"]) for x in audits]
producer_wait = [int(x["w4u8_mlp_producer_slot_wait_ticks"]) for x in audits]
mean_wait = [wait / claims for wait, claims in zip(wait_sums, claim_counts)]
mean_ahead = [ahead / claims for ahead, claims in zip(ahead_sums, claim_counts)]
mean_activation_work = [work / claims for work, claims in zip(activation_work, claim_counts)]

ordinary_instrumentation_off = all(
    int(ordinary[key]) == 0
    for key in (
        "w4u8_mlp_activation_queue_claim_count",
        "w4u8_mlp_activation_queue_wait_ticks",
        "w4u8_mlp_activation_queue_tasks_ahead_sum",
    )
)
correctness_pass = (
    int(correctness.get("mismatch_count", correctness.get("mismatches", 0))) == 0
    and int(correctness.get("max_lsb", 0)) == 0
    and int(correctness.get("rpc_result", -1)) == 0
    and int(correctness.get("dsp_status", -1)) == 3
)
physical_pass = (
    int(ordinary.get("vtcm_requested_bytes", 0)) == 8388608
    and int(ordinary.get("vtcm_acquired_bytes", 0)) == 8388608
    and int(ordinary.get("intermediate_ddr_read_bytes", 0)) == 0
    and int(ordinary.get("intermediate_ddr_write_bytes", 0)) == 0
    and int(ordinary.get("intermediate_spill_fill_count", 0)) == 0
    and int(ordinary.get("hmx_command_count", 0)) == 176
    and int(ordinary.get("hmx_u8s8_tile_pair_count", 0)) == 49408
)
residency_repeatable = sum(
    wait > work and ahead >= 4
    for wait, work, ahead in zip(wait_maxima, mean_activation_work, ahead_maxima)
) >= 6
material_residency = (
    med(mean_wait) > med(mean_activation_work)
    and med(ahead_maxima) >= 4
    and med(producer_wait) > 0
)
gate_pass = all(claim == 192 for claim in claim_counts) and (
    ordinary_instrumentation_off
    and correctness_pass
    and physical_pass
    and residency_repeatable
    and material_residency
)

summary = {
    "experiment": "EXP-0090",
    "stage": "A",
    "audit_runs": len(audits),
    "activation_claim_count_min_max": [min(claim_counts), max(claim_counts)],
    "activation_queue_mean_wait_ticks_median": med(mean_wait),
    "activation_queue_max_wait_ticks_median": med(wait_maxima),
    "activation_tasks_ahead_mean_median": med(mean_ahead),
    "activation_tasks_ahead_max_median": med(ahead_maxima),
    "queue_depth_max_median": med(depth_maxima),
    "activation_work_per_task_ticks_median": med(mean_activation_work),
    "producer_slot_wait_ticks_median": med(producer_wait),
    "ordinary_instrumentation_off": ordinary_instrumentation_off,
    "correctness_gate": correctness_pass,
    "physical_gate": physical_pass,
    "residency_repeatable": residency_repeatable,
    "material_residency": material_residency,
    "gate_pass": gate_pass,
    "decision": "proceed_to_stage_b" if gate_pass else "stop_before_stage_b",
}
print(json.dumps(summary, indent=2, sort_keys=True))
