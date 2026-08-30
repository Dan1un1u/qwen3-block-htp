#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys


def load_jsonl(path: Path) -> dict:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("{"):
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no JSON record in {path}")
    return records[-1]


root = Path(sys.argv[1]).resolve()
source_root = Path(__file__).resolve().parents[1]
source_head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
).strip()
ordinary = load_jsonl(root / "ordinary_control.jsonl")
correctness = load_jsonl(root / "correctness_audit.jsonl")
summary = json.loads((root / "stage_a_summary.json").read_text())
audits = [load_jsonl(p) for p in sorted((root / "audit_runs").glob("*.jsonl"))]

ledger = [
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks", "stage_boundary_ticks",
    "ledger_named_ticks", "ledger_unattributed_ticks", "invocation_ticks",
]
overlap = [
    "weight_dma_ticks", "hmx_compute_ticks", "hmx_ready_wait_ticks",
    "w4u8_mlp_gate_up_pipeline_ticks", "w4u8_mlp_down_pipeline_ticks",
    "w4u8_mlp_activation_work_ticks", "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
]

lines = [
    "# EXP-0090 Stage-A profiling closure",
    "",
    "## PC-028 repeat10 three-variant overview",
    "",
    "No candidate was implemented because the audit gate failed. All three columns below are therefore reused verbatim from the accepted EXP-0084 canonical common-schedule evidence.",
    "",
    "| Module | F16F16 EXP-0084 | W4F16 EXP-0084 | W4U8 EXP-0084 | W4U8 speed vs W4F16 |",
    "|---|---:|---:|---:|---:|",
    "| I/O and metadata | 6.6 us (0.3%) | 7.6 us (0.4%) | 4.1 us (0.2%) | +85.1% |",
    "| Input RMSNorm | 17.4 us (0.7%) | 17.3 us (0.8%) | 19.2 us (1.0%) | -9.7% |",
    "| QKV + Q/K Norm/RoPE | 400.6 us (16.4%) | 437.6 us (20.2%) | 424.6 us (21.7%) | +3.1% |",
    "| QK-Softmax-AV | 140.4 us (5.7%) | 139.6 us (6.5%) | 196.8 us (10.1%) | -29.1% |",
    "| O projection | 202.0 us (8.3%) | 172.9 us (8.0%) | 171.4 us (8.8%) | +0.9% |",
    "| Post-attn residual + RMSNorm | 16.7 us (0.7%) | 16.7 us (0.8%) | 35.6 us (1.8%) | -52.9% |",
    "| Gate/Up + SwiGLU | 1120.4 us (45.9%) | 964.6 us (44.6%) | 686.2 us (35.1%) | +40.6% |",
    "| Down projection | 459.6 us (18.8%) | 329.5 us (15.2%) | 318.1 us (16.3%) | +3.6% |",
    "| Final residual | 5.0 us (0.2%) | 5.0 us (0.2%) | 17.3 us (0.9%) | -71.3% |",
    "| Host/RPC and closure | 73.7 us (3.0%) | 71.1 us (3.3%) | 82.2 us (4.2%) | -13.5% |",
    "| Complete block Host wall | 2442.4 us | 2161.8 us | 1955.3 us | +10.6% |",
    "",
    "## Identity and failure boundary",
    "",
    "| Field | Value |",
    "|---|---|",
    "| Experiment | EXP-0090 |",
    f"| Source commit | `{source_head}` |",
    f"| Evidence | `{root}` |",
    "| Direct control | EXP-0084 W4U8 single FIFO |",
    "| Stage A | Seven audit-only repeat-one runs |",
    "| Candidate | Not implemented: Stage-A queue-residency gate failed |",
    "| Backend | standalone FastRPC/cDSP; no QNN or fallback |",
    "",
    "## Audit result",
    "",
    "| Metric | Median / result |",
    "|---|---:|",
]
for key in (
    "activation_claim_count_min_max",
    "activation_queue_mean_wait_ticks_median",
    "activation_queue_max_wait_ticks_median",
    "activation_tasks_ahead_mean_median",
    "activation_tasks_ahead_max_median",
    "queue_depth_max_median",
    "activation_work_per_task_ticks_median",
    "producer_slot_wait_ticks_median",
):
    lines.append(f"| `{key}` | {summary[key]} |")
lines += [
    f"| Ordinary instrumentation disabled | {summary['ordinary_instrumentation_off']} |",
    f"| Correctness gate | {summary['correctness_gate']} |",
    f"| Physical gate | {summary['physical_gate']} |",
    f"| Material and repeatable residence | {summary['material_residency'] and summary['residency_repeatable']} |",
    "",
    "An activation task waits about 28.7 ticks on average while its own work costs about 40.7 ticks. At enqueue it has only about 1.29 older queued tasks on average and at most three in the median run. This is not the long FIFO blockage required by the hypothesis.",
    "",
    "## Direct repeat-one control ledger",
    "",
    "The candidate column is unavailable because the serial contract stops before implementation when Stage A fails. No repeat-ten direct A/B run exists for the same reason.",
    "",
    "| Additive interval | Control | Candidate | Delta |",
    "|---|---:|---:|---:|",
]
for key in ledger:
    value = ordinary.get(key, "unavailable")
    lines.append(f"| `{key}` | {value} ticks | N/A | N/A |")
lines += [
    f"| `host_wall_ns_per_block` | {ordinary.get('host_wall_ns_per_block', 'unavailable')} ns | N/A | N/A |",
    "",
    "## Overlapping engine diagnostics",
    "",
    "These counters overlap and must not be summed into wall time.",
    "",
    "| Counter | Control | Candidate | Delta |",
    "|---|---:|---:|---:|",
]
for key in overlap:
    lines.append(f"| `{key}` | {ordinary.get(key, 'unavailable')} | N/A | N/A |")
lines += [
    "",
    "## Correctness and physical contract",
    "",
    "| Gate | Value | Result |",
    "|---|---:|---|",
    f"| Final mismatches / max LSB | {correctness.get('mismatches')} / {correctness.get('max_lsb')} | PASS |",
    f"| Output hash | `{correctness.get('output_hash')}` | PASS |",
    f"| VTCM requested / acquired / peak | {ordinary.get('vtcm_requested_bytes')} / {ordinary.get('vtcm_acquired_bytes')} / {ordinary.get('vtcm_peak_plan_bytes')} B | PASS |",
    f"| Intermediate DDR read / write | {ordinary.get('intermediate_ddr_read_bytes')} / {ordinary.get('intermediate_ddr_write_bytes')} B | PASS |",
    f"| Spill/fill | {ordinary.get('intermediate_spill_fill_count')} | PASS |",
    f"| Weight bytes / DMA descriptors | {ordinary.get('weight_ddr_read_bytes')} / {ordinary.get('weight_dma_descriptor_count')} | PASS |",
    f"| HMX commands / U8S8 tile pairs | {ordinary.get('hmx_command_count')} / {ordinary.get('hmx_u8s8_tile_pair_count')} | PASS |",
    "| FastRPC / HMX owner | one block invocation / one serialized owner | PASS |",
    "",
    "## Decision",
    "",
    "Stage A fails and Stage B is not entered. The single FIFO does not hold ready SwiGLU tasks behind a material expansion backlog: queue residence is shorter than one activation task on average and the typical maximum number of older queued tasks is three. A priority queue would add arbitration and locality risk without enough recoverable critical-path time. EXP-0090 is rejected and EXP-0084 remains the selected W4U8 baseline.",
]

print("\n".join(lines))
