#!/usr/bin/env python3
"""Validate and report EXP-0091 Gate/Up interleaved VLUT decoding."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SAMPLES = 5
REPEATS = (1, 10)
MODES = ("control", "candidate")
OUTPUT_HASH = "69f22eeb035e5ec5"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
VTCM_BYTES = 8_388_608
WEIGHT_BYTES = 25_444_352
DMA_DESCRIPTORS = 512
HMX_COMMANDS = 176
HMX_TILE_PAIRS = 49_408

LEDGER_FIELDS = (
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks",
    "stage_boundary_ticks", "invocation_ticks", "ledger_named_ticks",
    "ledger_unattributed_ticks",
)
OVERLAP_FIELDS = (
    "w4u8_mlp_gate_up_pipeline_ticks", "w4u8_mlp_down_pipeline_ticks",
    "w4u8_mlp_activation_work_ticks", "w4u8_mlp_weight_stage_ticks",
    "w4u8_mlp_weight_expand_ticks", "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_hmx_ready_wait_ticks",
    "w4u8_mlp_producer_slot_wait_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks", "weight_dma_ticks",
    "hmx_compute_ticks", "hmx_ready_wait_ticks",
)
INVARIANT_FIELDS = (
    "vtcm_requested_bytes", "vtcm_acquired_bytes",
    "vtcm_peak_plan_bytes", "weight_ddr_read_bytes",
    "weight_dma_descriptor_count", "hmx_command_count",
    "hmx_u8s8_tile_pair_count", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count", "w4u8_mlp_gate_up_hmx_command_count",
    "w4u8_mlp_down_hmx_command_count",
)


def load(path: Path, expected: int) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if len(records) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {len(records)}")
    return records


def require(record: dict[str, object], field: str, expected: object) -> None:
    if record.get(field) != expected:
        raise SystemExit(
            f"{field}: expected {expected!r}, got {record.get(field)!r}"
        )


def validate(record: dict[str, object], repeat: int, mode: str) -> None:
    require(record, "experiment", "EXP-0091")
    require(record, "variant", "W4U8")
    require(record, "w4u8_gate_up_decode_mode",
            "control" if mode == "control" else "interleaved2")
    require(record, "output_hash", OUTPUT_HASH)
    require(record, "mismatches", 0)
    require(record, "max_lsb", 0)
    require(record, "dsp_status", 3)
    require(record, "numerical_status", 1)
    require(record, "vtcm_requested_bytes", VTCM_BYTES)
    require(record, "vtcm_acquired_bytes", VTCM_BYTES)
    require(record, "intermediate_ddr_read_bytes", 0)
    require(record, "intermediate_ddr_write_bytes", 0)
    require(record, "intermediate_dma_descriptor_count", 0)
    require(record, "intermediate_spill_fill_count", 0)
    require(record, "weight_ddr_read_bytes", WEIGHT_BYTES * repeat)
    require(record, "weight_dma_descriptor_count",
            DMA_DESCRIPTORS * repeat)
    require(record, "hmx_command_count", HMX_COMMANDS * repeat)
    require(record, "hmx_u8s8_tile_pair_count", HMX_TILE_PAIRS * repeat)
    require(record, "block_invocation_count", repeat)


def summarize(control: list[dict[str, object]],
              candidate: list[dict[str, object]], field: str,
              divisor: float = 1.0) -> dict[str, float | None]:
    left = [float(record[field]) / divisor for record in control]
    right = [float(record[field]) / divisor for record in candidate]
    control_median = statistics.median(left)
    candidate_median = statistics.median(right)
    if control_median == 0.0 or any(value == 0.0 for value in left):
        return {
            "control": control_median, "candidate": candidate_median,
            "change_percent": None, "paired_change_percent_median": None,
        }
    paired = [(r / l - 1.0) * 100.0 for l, r in zip(left, right)]
    return {
        "control": control_median,
        "candidate": candidate_median,
        "change_percent": (candidate_median / control_median - 1.0) * 100.0,
        "paired_change_percent_median": statistics.median(paired),
    }


def build_summary(root: Path) -> dict[str, object]:
    if (root / "boot_id_before.txt").read_bytes() != (
            root / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed")
    static_gate = json.loads((root / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")

    correctness: dict[str, object] = {}
    for mode in MODES:
        record = load(root / f"correctness_{mode}.jsonl", 1)[0]
        validate(record, 1, mode)
        require(record, "u8_attention_actual_score_hash", QK_HASH)
        require(record, "u8_attention_actual_probability_hash",
                PROBABILITY_HASH)
        require(record, "u8_attention_actual_av_hash", AV_HASH)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
            "qk_hash": record["u8_attention_actual_score_hash"],
            "probability_hash": record[
                "u8_attention_actual_probability_hash"],
            "av_hash": record["u8_attention_actual_av_hash"],
        }

    repeat_results: dict[str, object] = {}
    gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load(root / f"paired_{mode}_r{repeat}.jsonl", SAMPLES)
            for mode in MODES
        }
        for mode in MODES:
            for record in records[mode]:
                validate(record, repeat, mode)
        metrics = {
            field: summarize(records["control"], records["candidate"],
                             field,
                             1.0 if field == "host_wall_ns_per_block"
                             else float(repeat))
            for field in (
                "host_wall_ns_per_block", *LEDGER_FIELDS, *OVERLAP_FIELDS,
                *INVARIANT_FIELDS,
            )
        }
        gate_up = metrics["gate_up_ticks"]
        host = metrics["host_wall_ns_per_block"]
        speed_gate = all(
            metric[key] is not None and float(metric[key]) < 0.0
            for metric in (gate_up, host)
            for key in ("change_percent", "paired_change_percent_median")
        )
        invariant_gate = all(
            all(record[field] == records["control"][index][field]
                for index, record in enumerate(records["candidate"]))
            for field in INVARIANT_FIELDS
        )
        repeat_gate = speed_gate and invariant_gate
        gates.append(repeat_gate)
        repeat_results[f"repeat{repeat}"] = {
            "metrics": metrics,
            "strict_gate_up_and_host_speed_gate": speed_gate,
            "unchanged_physical_contract_gate": invariant_gate,
            "repeat_gate_pass": repeat_gate,
        }
    return {
        "experiment": "EXP-0091",
        "stage": "B",
        "samples_per_cell": SAMPLES,
        "static_gate": static_gate,
        "correctness": correctness,
        "physical_gate": {
            "vtcm_requested_and_acquired_bytes": VTCM_BYTES,
            "intermediate_ddr_bytes": 0,
            "intermediate_spill_fill_count": 0,
            "weight_ddr_bytes_per_block": WEIGHT_BYTES,
            "weight_dma_descriptors_per_block": DMA_DESCRIPTORS,
            "hmx_commands_per_block": HMX_COMMANDS,
            "u8s8_tile_pairs_per_block": HMX_TILE_PAIRS,
            "hmx_owners": 1, "qnn": False, "pass": True,
        },
        "repeat_results": repeat_results,
        "stage_b_gate_pass": all(gates),
    }


def metric_row(name: str, metric: dict[str, object], unit: str) -> str:
    return (
        f"| {name} | {float(metric['control']):.3f} {unit} | "
        f"{float(metric['candidate']):.3f} {unit} | "
        f"{float(metric['change_percent']):+.3f}% | "
        f"{float(metric['paired_change_percent_median']):+.3f}% |"
    )


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0091 — Complete profiling report", "",
        "The candidate changes only Gate/Up packed-W4 nibble decoding from "
        "one-vector VLUT32 scheduling to two independent interleaved vectors. "
        "Quantization, bytes, tasks, DMA, HMX and tensor boundaries are fixed.",
        "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        metrics = result["metrics"]
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Primary target | Control | Candidate | Ordinary change | Paired median change |",
            "|---|---:|---:|---:|---:|",
            metric_row("Complete block Host wall",
                       metrics["host_wall_ns_per_block"], "ns"),
            metric_row("Gate/Up", metrics["gate_up_ticks"], "ticks"),
            metric_row("MLP W4 expansion",
                       metrics["w4u8_mlp_weight_expand_ticks"], "ticks"),
            metric_row("MLP HMX ready wait",
                       metrics["w4u8_mlp_hmx_ready_wait_ticks"], "ticks"),
            "",
            f"Strict Gate/Up + Host speed gate: **{'PASS' if result['strict_gate_up_and_host_speed_gate'] else 'FAIL'}**. "
            f"Physical invariants: **{'PASS' if result['unchanged_physical_contract_gate'] else 'FAIL'}**.",
            "", "### Additive block timing ledger", "",
            "| Field | Control ticks/block | Candidate ticks/block | Change | Paired change |",
            "|---|---:|---:|---:|---:|",
        ])
        for field in LEDGER_FIELDS:
            metric = metrics[field]
            if metric["change_percent"] is None:
                continue
            lines.append(metric_row(field, metric, "ticks"))
        lines.extend([
            "", "### Overlapping engine work and waits", "",
            "| Field | Control ticks/block | Candidate ticks/block | Change | Paired change |",
            "|---|---:|---:|---:|---:|",
        ])
        for field in OVERLAP_FIELDS:
            metric = metrics[field]
            if metric["change_percent"] is None:
                continue
            lines.append(metric_row(field, metric, "ticks"))

    lines.extend([
        "", "## PC-028 stable accepted three-recipe repeat10 table", "",
        "This table deliberately remains the accepted EXP-0084 comparison; EXP-0091 is not a baseline unless the user promotes it.", "",
        "| Module | W16A16 | W4A16 | W4A8 accepted | A8 vs A16 speed |",
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
        "| Host/RPC and closure remainder | 73.7 us (3.0%) | 71.1 us (3.3%) | 82.2 us (4.2%) | -13.5% |",
        "| Complete block Host wall | 2442.4 us | 2161.8 us | 1955.3 us | +10.6% |",
        "", "## Correctness and physical gates", "",
        "Final output and audited Attention boundaries are byte-exact to EXP-0084 with zero mismatch and zero maximum LSB. Requested/acquired VTCM is exactly 8 MiB; intermediate DDR and spill/fill are zero; the block remains one FastRPC with one HMX owner and no QNN.",
        "", "## Decision", "",
        f"EXP-0091 Stage-B local gate: **{'PASS' if summary['stage_b_gate_pass'] else 'FAIL'}**. The accepted baseline remains EXP-0084 unless the user explicitly promotes a passing candidate.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    summary = build_summary(args.result_dir)
    print(render_report(summary) if args.report else
          json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
