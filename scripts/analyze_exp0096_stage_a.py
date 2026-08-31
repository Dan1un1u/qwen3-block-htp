#!/usr/bin/env python3
"""Validate the EXP-0096 correctness stop boundary and render closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import statistics
from pathlib import Path

import validate_exp0050 as base
import validate_exp0084 as exp84


OUTPUT_HASH = "69f22eeb035e5ec5"
INPUT_NORM_HASH = "7255c2406108617c"
QK_HASH = "32aa949912e365be"
PROBABILITY_HASH = "94f2e218f06f9627"
AV_HASH = "f853658f52032bde"
EXP0084_EVIDENCE = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0084/"
    "20260830T160259Z_6dc437fe08ea_formal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def load_one(path: Path) -> dict[str, object]:
    records = base.load_jsonl(path, 1)
    return records[0]


def require_physical(record: dict[str, object], arithmetic_tasks: int) -> None:
    expected = {
        "experiment": "EXP-0096",
        "variant": "W4U8",
        "repeat_count": 1,
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "vtcm_peak_plan_bytes": 5_306_080,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "weight_ddr_read_bytes": 25_444_352,
        "weight_dma_descriptor_count": 512,
        "hmx_command_count": 176,
        "hmx_u8s8_tile_pair_count": 49_408,
        "w4u8_mlp_lut_vtcm_bytes": 131_072,
        "w4u8_mlp_gather_scratch_vtcm_bytes": 768,
        "w4u8_mlp_pair_publish_count": 192,
        "w4u8_mlp_pair_consume_count": 192,
        "w4u8_mlp_arithmetic_activation_task_count": arithmetic_tasks,
    }
    for field, value in expected.items():
        base.require(record, field, value)
    boundaries = (
        record.get("u8_input_norm_actual_hash"),
        record.get("u8_attention_actual_score_hash"),
        record.get("u8_attention_actual_probability_hash"),
        record.get("u8_attention_actual_av_hash"),
    )
    if boundaries != (INPUT_NORM_HASH, QK_HASH, PROBABILITY_HASH, AV_HASH):
        raise SystemExit(f"unexpected audited boundary hashes: {boundaries!r}")


def floor_div_pow2(value: int, shift: int) -> int:
    if value >= 0:
        return value >> shift
    return -(((-value) + ((1 << shift) - 1)) >> shift)


def old_direct_value(gate: int, up: int) -> int:
    x = gate - 128
    numerator = max(0, min(128, x + 64))
    silu = floor_div_pow2(x * numerator + 64, 7)
    centered = floor_div_pow2(silu * (up - 128) + 16, 5)
    return max(0, min(255, centered + 128))


def full_domain_formula_audit(package_dir: Path) -> dict[str, int]:
    raw = (package_dir / "silu_up_lut_u16.bin").read_bytes()
    if len(raw) != 131_072:
        raise SystemExit("unexpected activation LUT size")
    mismatches = 0
    maximum = 0
    for gate in range(256):
        for up in range(256):
            expected = struct.unpack_from(
                "<H", raw, 2 * (gate * 256 + up)
            )[0]
            actual = old_direct_value(gate, up)
            error = abs(actual - expected)
            mismatches += error != 0
            maximum = max(maximum, error)
    return {"mismatches": mismatches, "max_lsb": maximum}


def module_medians(variant: str) -> dict[str, float]:
    records = exp84.load_jsonl(
        EXP0084_EVIDENCE / "tri_variant" / "canonical" /
        f"{variant}_r10.jsonl", 7
    )
    samples = [dict(exp84.module_us(record)) for record in records]
    return {
        name: float(statistics.median(sample[name] for sample in samples))
        for name in samples[0]
    }


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != \
            (result_dir / "boot_id_after.txt").read_bytes():
        raise SystemExit("device boot ID changed during EXP-0096 stage A")
    static_gate = json.loads((result_dir / "static_gate.json").read_text())
    if static_gate.get("static_gate") != "pass":
        raise SystemExit("static gate failed")
    control = load_one(result_dir / "correctness_control.jsonl")
    candidate = load_one(result_dir / "correctness_candidate.jsonl")
    require_physical(control, 0)
    require_physical(candidate, 192)
    if control.get("mlp_mode") != \
            "w4u8_streaming_persistent_mlp_hvx":
        raise SystemExit("wrong control MLP mode")
    if candidate.get("mlp_mode") != \
            "w4u8_streaming_persistent_mlp_hvx_arithmetic_activation":
        raise SystemExit("wrong candidate MLP mode")
    if control.get("output_hash") != OUTPUT_HASH or \
            int(control.get("mismatches", -1)) != 0 or \
            int(control.get("max_lsb", -1)) != 0:
        raise SystemExit("control is not byte-exact")
    candidate_mismatches = int(candidate.get("mismatches", -1))
    candidate_max_lsb = int(candidate.get("max_lsb", -1))
    if candidate_mismatches <= 0:
        raise SystemExit("candidate unexpectedly passed the failure boundary")
    formula = full_domain_formula_audit(package_dir)
    fields = (
        "w4u8_mlp_activation_work_ticks",
        "gate_up_ticks",
        "host_wall_ns_per_block",
    )
    diagnostic = {
        field: {
            "control": float(control[field]),
            "candidate": float(candidate[field]),
            "change_percent": (
                float(candidate[field]) / float(control[field]) - 1.0
            ) * 100.0,
        }
        for field in fields
    }
    return {
        "experiment": "EXP-0096",
        "source_commit": (result_dir / "source_commit.txt").read_text().strip(),
        "package_manifest_sha256": hashlib.sha256(
            (package_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "static_gate": static_gate,
        "physical_gate": "pass",
        "control_byte_exact_gate": "pass",
        "candidate_correctness_gate": "fail",
        "candidate_output_hash": candidate.get("output_hash"),
        "candidate_mismatches": candidate_mismatches,
        "candidate_max_lsb": candidate_max_lsb,
        "full_domain_old_direct_vs_formal_lut": formula,
        "diagnostic_only_single_run": diagnostic,
        "speed_gate": "not_run_by_contract_after_correctness_failure",
        "local_gate_pass": False,
        "adoption_status": "rejected",
        "pc028": {
            "f16f16": module_medians("f16f16"),
            "w4f16": module_medians("w4f16"),
            "w4u8": module_medians("w4u8"),
            "provenance": str(EXP0084_EVIDENCE),
        },
    }


def add_pc028(lines: list[str], summary: dict[str, object]) -> None:
    table = summary["pc028"]
    f16 = table["f16f16"]
    w4f16 = table["w4f16"]
    w4u8 = table["w4u8"]
    totals = {
        "f16": f16["Complete block Host wall"],
        "w4f16": w4f16["Complete block Host wall"],
        "w4u8": w4u8["Complete block Host wall"],
    }
    lines.extend([
        "## PC-028 accepted-baseline repeat-ten module table", "",
        "EXP-0096 stops at correctness, so all three columns reuse the "
        "accepted EXP-0084 canonical evidence.", "",
        "| Module | W16A16 | W4A16 | W4A8 | A8 vs A16 speed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in f16:
        if name == "Complete block Host wall":
            values = (f"{f16[name]:.1f} us", f"{w4f16[name]:.1f} us",
                      f"{w4u8[name]:.1f} us")
        else:
            values = (
                f"{f16[name]:.1f} us ({100*f16[name]/totals['f16']:.1f}%)",
                f"{w4f16[name]:.1f} us ({100*w4f16[name]/totals['w4f16']:.1f}%)",
                f"{w4u8[name]:.1f} us ({100*w4u8[name]/totals['w4u8']:.1f}%)",
            )
        speed = (w4f16[name] / w4u8[name] - 1.0) * 100.0
        lines.append(
            f"| {name} | {values[0]} | {values[1]} | {values[2]} | "
            f"{speed:+.1f}% |"
        )
    lines.append("")


def render_report(summary: dict[str, object]) -> str:
    lines = ["# EXP-0096 — Correctness failure boundary", ""]
    add_pc028(lines, summary)
    diagnostic = summary["diagnostic_only_single_run"]
    lines.extend([
        "## Stage-A evidence", "",
        "The candidate reaches the intended zero-vgather HVX multiply/shift/"
        "pack kernel and preserves the complete physical contract, but it is "
        "not the same mathematical mapping as the formal 128 KiB LUT.", "",
        "| Gate | Result |", "|---|---:|",
        "| Static schedule | PASS: 0 vgather, 4 vmpy, 5 shift/pack |",
        "| VTCM / intermediate DDR | 8 MiB / 0 read, 0 write |",
        "| HMX physical work | unchanged: 176 commands, 49,408 tile pairs |",
        f"| Candidate final output | FAIL: {summary['candidate_mismatches']} "
        f"mismatches, {summary['candidate_max_lsb']} max LSB |",
        f"| Full 256x256 mapping | FAIL: "
        f"{summary['full_domain_old_direct_vs_formal_lut']['mismatches']} "
        f"mismatches, {summary['full_domain_old_direct_vs_formal_lut']['max_lsb']} "
        "max LSB |", "",
        "The one-run timing below is diagnostic only; no performance gate is "
        "claimed after correctness failed.", "",
        "| Metric | Control | Candidate | Change |", "|---|---:|---:|---:|",
    ])
    for field, metric in diagnostic.items():
        lines.append(
            f"| `{field}` | {metric['control']:.3f} | "
            f"{metric['candidate']:.3f} | {metric['change_percent']:+.3f}% |"
        )
    lines.extend([
        "", "## Decision", "",
        "EXP-0096 is **rejected at the correctness gate**. The accepted "
        "EXP-0084 baseline remains unchanged. A compact one-dimensional "
        "coefficient representation would be a different hypothesis and "
        "must not be reported as this experiment.", "",
        f"Source commit: `{summary['source_commit']}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.result_dir, args.package_dir)
    if args.report:
        print(render_report(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
