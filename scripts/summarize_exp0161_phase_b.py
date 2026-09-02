#!/usr/bin/env python3
"""Summarize EXP-0161 Phase-B segmented-cache evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


TICKS_PER_US = 19.2
LENGTHS = (64, 256, 1024, 4096)
REPEATS = (1, 10)
MODES = ("control", "candidate")
HEADS = 8
HEAD_DIM = 128
BLOCK_M = 64
SEGMENT_TOKENS = 32
HMX_BIAS_BYTES = 256
HMX_WEIGHT_BYTES = 1024
HEAD_DIM_TILES = 4
MODEL_ROOT = Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0161")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def read_json_record(path: Path) -> dict[str, object] | None:
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("experiment") == "EXP-0161":
            return value
    return None


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def current_cache_row(data: bytes, length: int, mode: str, kind: str) -> bytes:
    capacity = length + 1
    if mode == "control":
        delta_rows = capacity - BLOCK_M
        bias_bytes = 2 * HMX_BIAS_BYTES if kind == "k" else HEAD_DIM_TILES * HMX_BIAS_BYTES
        head_bytes = BLOCK_M * HEAD_DIM + bias_bytes + delta_rows * HEAD_DIM
        row_offset = BLOCK_M * HEAD_DIM + bias_bytes + (length - BLOCK_M) * HEAD_DIM
    else:
        segments = length // SEGMENT_TOKENS
        if kind == "k":
            head_bytes = segments * (SEGMENT_TOKENS * HEAD_DIM + HMX_BIAS_BYTES) + SEGMENT_TOKENS * HEAD_DIM
            row_offset = segments * (SEGMENT_TOKENS * HEAD_DIM + HMX_BIAS_BYTES)
        else:
            head_bytes = segments * SEGMENT_TOKENS * HEAD_DIM + HEAD_DIM_TILES * HMX_BIAS_BYTES + SEGMENT_TOKENS * HEAD_DIM
            row_offset = segments * SEGMENT_TOKENS * HEAD_DIM + HEAD_DIM_TILES * HMX_BIAS_BYTES
    rows = []
    for head in range(HEADS):
        first = head * head_bytes + row_offset
        rows.append(data[first:first + HEAD_DIM])
    return b"".join(rows)


def reference_cache_row(data: bytes, length: int) -> bytes:
    capacity = length + 1
    head_bytes = capacity * HEAD_DIM
    row_offset = length * HEAD_DIM
    return b"".join(
        data[head * head_bytes + row_offset:
             head * head_bytes + row_offset + HEAD_DIM]
        for head in range(HEADS)
    )


def difference_metrics(actual: bytes, expected: bytes) -> tuple[int, int]:
    if len(actual) != len(expected):
        raise ValueError(
            f"comparison size mismatch: actual={len(actual)}, expected={len(expected)}"
        )
    differences = [abs(left - right) for left, right in zip(actual, expected)]
    return sum(value != 0 for value in differences), max(differences, default=0)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    records: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    for path in sorted((result_dir / "raw").glob("*.jsonl")):
        parts = path.stem.split("_")
        length = int(parts[0][1:])
        repeat_count = int(parts[1][1:])
        mode = parts[-1]
        status = int(Path(str(path) + ".status").read_text().strip())
        record = read_json_record(path)
        if record is None or status != 0:
            failures.append({
                "length": length,
                "repeat": repeat_count,
                "mode": mode,
                "status": status,
                "stderr": str(path.with_suffix(".stderr")),
            })
        else:
            records[(length, repeat_count, mode)].append(record)

    cells: dict[str, dict[str, object]] = {}
    for length in LENGTHS:
        for repeat_count in REPEATS:
            for mode in MODES:
                runs = records.get((length, repeat_count, mode), [])
                name = f"l{length}_r{repeat_count}_{mode}"
                if not runs:
                    cells[name] = {"available": False}
                    continue
                divisor = float(repeat_count)
                physical = all(
                    int(run["rpc_result"]) == 0
                    and int(run["dsp_status"]) == 3
                    and int(run["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(run["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(run["intermediate_ddr_read_bytes"]) == 0
                    and int(run["intermediate_ddr_write_bytes"]) == 0
                    and int(run["intermediate_spill_fill_count"]) == 0
                    and int(run["ledger_unattributed_ticks"]) == 0
                    and int(run["block_invocation_count"]) == repeat_count
                    and run["intermediate_residency"] == "VTCM"
                    for run in runs
                )
                cells[name] = {
                    "available": True,
                    "runs": len(runs),
                    "host_us_per_block": median([
                        float(run["host_wall_ns_per_block"]) / 1000.0
                        for run in runs
                    ]),
                    "attention_us_per_block": median([
                        float(run["attention_ticks"]) / divisor / TICKS_PER_US
                        for run in runs
                    ]),
                    "cache_stage_us_per_block": median([
                        float(run["scan_cache_stage_ticks"]) / divisor / TICKS_PER_US
                        for run in runs
                    ]),
                    "cache_descriptors_per_block": median([
                        float(run["scan_cache_dma_descriptor_count"]) / divisor
                        for run in runs
                    ]),
                    "cache_read_bytes_per_block": median([
                        float(run["scan_cache_ddr_read_bytes"]) / divisor
                        for run in runs
                    ]),
                    "overlay_required_bytes": int(runs[0]["scan_attention_overlay_required_bytes"]),
                    "overlay_capacity_bytes": int(runs[0]["scan_attention_overlay_capacity_bytes"]),
                    "physical_pass": physical,
                    "max_lsb_to_inherited_reference": max(int(run["max_lsb"]) for run in runs),
                    "output_hashes": sorted({str(run["output_hash"]) for run in runs}),
                    "full_prefix_pack_count": max(int(run.get("u8_cache_full_prefix_pack_count", 0)) for run in runs),
                }

    speed: dict[str, dict[str, float | bool]] = {}
    for length in LENGTHS:
        control = cells[f"l{length}_r10_control"]
        candidate = cells[f"l{length}_r10_candidate"]
        control_us = float(control["host_us_per_block"])
        candidate_us = float(candidate["host_us_per_block"])
        improvement = (control_us / candidate_us - 1.0) * 100.0
        speed[f"l{length}"] = {
            "control_us": control_us,
            "candidate_us": candidate_us,
            "improvement_percent": improvement,
            "pass": improvement >= 0.0 if length == 64 else improvement > 0.0,
        }

    capture: dict[str, dict[str, object]] = {}
    for length in LENGTHS:
        roots = {
            mode: result_dir / "capture" / f"l{length}_{mode}"
            for mode in MODES
        }
        control_output = (roots["control"] / "block_output_u8.bin").read_bytes()
        candidate_output = (roots["candidate"] / "block_output_u8.bin").read_bytes()
        logical_control = control_output[:2048]
        logical_candidate = candidate_output[:2048]
        reference_root = (
            MODEL_ROOT / f"decode_l{length}_w4u8_hmx_segmented_v4b"
        )
        logical_reference = (
            reference_root / "reference_exp0147_cpu_block_output_u8.bin"
        ).read_bytes()[:2048]
        candidate_cpu_mismatches, candidate_cpu_max_lsb = difference_metrics(
            logical_candidate, logical_reference
        )
        control_cpu_mismatches, control_cpu_max_lsb = difference_metrics(
            logical_control, logical_reference
        )
        pair_mismatches, pair_max_lsb = difference_metrics(
            logical_candidate, logical_control
        )
        entry: dict[str, object] = {
            "candidate_cpu_output_mismatches": candidate_cpu_mismatches,
            "candidate_cpu_output_max_lsb": candidate_cpu_max_lsb,
            "control_cpu_output_mismatches": control_cpu_mismatches,
            "control_cpu_output_max_lsb": control_cpu_max_lsb,
            "candidate_control_output_mismatches": pair_mismatches,
            "candidate_control_output_max_lsb": pair_max_lsb,
            "control_output_sha256": digest(control_output),
            "candidate_output_sha256": digest(candidate_output),
            "cpu_reference_sha256": digest(logical_reference),
        }
        for kind in ("k", "v"):
            control_cache = (roots["control"] / f"actual_kv_cache_{kind}_u8.bin").read_bytes()
            candidate_cache = (roots["candidate"] / f"actual_kv_cache_{kind}_u8.bin").read_bytes()
            reference_cache = (
                reference_root / f"reference_kv_cache_{kind}_u8.bin"
            ).read_bytes()
            control_row = current_cache_row(control_cache, length, "control", kind)
            candidate_row = current_cache_row(candidate_cache, length, "candidate", kind)
            reference_row = reference_cache_row(reference_cache, length)
            for name, actual in (
                ("candidate", candidate_row),
                ("control", control_row),
            ):
                mismatches, max_lsb = difference_metrics(actual, reference_row)
                entry[f"{name}_current_{kind}_mismatches"] = mismatches
                entry[f"{name}_current_{kind}_max_lsb"] = max_lsb
        entry["pass"] = all(
            int(entry[key]) == 0
            for key in (
                "candidate_cpu_output_mismatches",
                "candidate_cpu_output_max_lsb",
                "candidate_current_k_mismatches",
                "candidate_current_k_max_lsb",
                "candidate_current_v_mismatches",
                "candidate_current_v_max_lsb",
            )
        )
        capture[f"l{length}"] = entry

    complete_cells = all(
        len(records.get((length, repeat_count, mode), [])) == 5
        for length in LENGTHS
        for repeat_count in REPEATS
        for mode in MODES
    )
    physical_pass = complete_cells and not failures and all(
        bool(cell.get("physical_pass"))
        for cell in cells.values() if cell.get("available")
    )
    speed_pass = all(bool(value["pass"]) for value in speed.values())
    correctness_pass = all(bool(value["pass"]) for value in capture.values())
    structural_pass = all(
        int(cells[f"l{length}_r10_candidate"]["full_prefix_pack_count"]) == 0
        and int(cells[f"l{length}_r10_candidate"]["overlay_required_bytes"])
            <= int(cells[f"l{length}_r10_candidate"]["overlay_capacity_bytes"])
        for length in LENGTHS
    ) and (
        int(cells["l1024_r10_candidate"]["overlay_required_bytes"])
        == int(cells["l4096_r10_candidate"]["overlay_required_bytes"])
    )

    summary = {
        "experiment": "EXP-0161",
        "phase": "B_segmented_cache_v4",
        "cells": cells,
        "speed_gate": speed,
        "capture_equivalence": capture,
        "physical_gate_pass": physical_pass,
        "correctness_gate_pass": correctness_pass,
        "structural_gate_pass": structural_pass,
        "speed_gate_pass": speed_pass,
        "unavailable_runs": failures,
        "reference_note": (
            "The correctness gate uses the retained independent EXP-0147 CPU "
            "integer block output and logical current-token K/V references. "
            "Candidate-versus-control differences are diagnostic only because "
            "long-KV HMX conversion can use a different rounding path."
        ),
    }
    (result_dir / "phase_b_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# EXP-0161 Phase B: segmented HMX-native W4U8 cache",
        "",
        "Each timing cell is the median of five rotated device runs. The speed gate uses repeat10 complete-block host wall time.",
        "",
        "| L | Control | Segmented V4 | Speed change | Control Attention | Segmented Attention | Candidate overlay |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length in LENGTHS:
        control = cells[f"l{length}_r10_control"]
        candidate = cells[f"l{length}_r10_candidate"]
        result = speed[f"l{length}"]
        lines.append(
            f"| {length} | {result['control_us']:.3f} us | "
            f"{result['candidate_us']:.3f} us | "
            f"{result['improvement_percent']:+.3f}% | "
            f"{control['attention_us_per_block']:.3f} us | "
            f"{candidate['attention_us_per_block']:.3f} us | "
            f"{candidate['overlay_required_bytes']}/{candidate['overlay_capacity_bytes']} B |"
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- Correctness (candidate vs independent CPU output and current K/V token, byte exact): {'PASS' if correctness_pass else 'FAIL'}",
        f"- Physical (8 MiB VTCM, zero intermediate DDR, zero spill/fill): {'PASS' if physical_pass else 'FAIL'}",
        f"- Structural (zero full-prefix pack and constant long-KV overlay): {'PASS' if structural_pass else 'FAIL'}",
        f"- Speed (L64 non-regression; L256/L1024/L4096 strictly faster): {'PASS' if speed_pass else 'FAIL'}",
        "",
        "The short path consumes at most one fixed 26-segment window; the long path uses the same 26-slot double bank at L1024 and L4096. No scratch allocation scales beyond that fixed window.",
        "",
        "The monolithic control's independent-reference and candidate/control differences are reported as diagnostics; they do not replace the candidate's independent correctness gate.",
    ]
    (result_dir / "phase_b_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result_dir": str(result_dir),
        "correctness_gate_pass": correctness_pass,
        "physical_gate_pass": physical_pass,
        "structural_gate_pass": structural_pass,
        "speed_gate_pass": speed_pass,
        "unavailable_runs": len(failures),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
