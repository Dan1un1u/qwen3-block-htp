#!/usr/bin/env python3
"""Summarize EXP-0161 Phase-A monolithic delta-cache length scaling."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


TICKS_PER_US = 19.2
LENGTHS = (64, 256, 1024, 4096)
REPEATS = (1, 10)
MODES = ("serial", "direct", "pipeline")


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


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


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
        if record is None:
            failures.append({
                "length": length,
                "repeat": repeat_count,
                "mode": mode,
                "status": status,
                "stderr": str(path.with_suffix(".stderr")),
            })
            continue
        records[(length, repeat_count, mode)].append(record)

    cells: dict[str, dict[str, object]] = {}
    for length in LENGTHS:
        for repeat_count in REPEATS:
            for mode in MODES:
                key = (length, repeat_count, mode)
                runs = records.get(key, [])
                name = f"l{length}_r{repeat_count}_{mode}"
                if not runs:
                    cells[name] = {"available": False}
                    continue
                per_block = float(repeat_count)
                output_hashes = {str(r["output_hash"]) for r in runs}
                k_hashes = {str(r["scan_cache_k_hash"]) for r in runs}
                v_hashes = {str(r["scan_cache_v_hash"]) for r in runs}
                physical = all(
                    int(r["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                    and int(r["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    and int(r["intermediate_ddr_read_bytes"]) == 0
                    and int(r["intermediate_ddr_write_bytes"]) == 0
                    and int(r["intermediate_spill_fill_count"]) == 0
                    and int(r["ledger_unattributed_ticks"]) == 0
                    and r["intermediate_residency"] == "VTCM"
                    for r in runs
                )
                cells[name] = {
                    "available": True,
                    "runs": len(runs),
                    "host_us_per_block": median([
                        float(r["host_wall_ns_per_block"]) / 1000.0 for r in runs
                    ]),
                    "attention_us_per_block": median([
                        float(r["attention_ticks"]) / per_block / TICKS_PER_US
                        for r in runs
                    ]),
                    "dynamic_attention_us_per_block": median([
                        float(r["scan_dynamic_attention_ticks"]) /
                        per_block / TICKS_PER_US for r in runs
                    ]),
                    "k_pack_us_per_block": median([
                        float(r["u8_attention_k_pack_ticks"]) /
                        per_block / TICKS_PER_US for r in runs
                    ]),
                    "v_pack_us_per_block": median([
                        float(r["u8_attention_v_pack_ticks"]) /
                        per_block / TICKS_PER_US for r in runs
                    ]),
                    "pipeline_wait_us_per_block": median([
                        float(r["u8_attention_pipeline_wait_ticks"]) /
                        per_block / TICKS_PER_US for r in runs
                    ]),
                    "cache_read_bytes_per_block": median([
                        float(r["scan_cache_ddr_read_bytes"]) / per_block
                        for r in runs
                    ]),
                    "cache_write_bytes_per_block": median([
                        float(r["scan_cache_ddr_write_bytes"]) / per_block
                        for r in runs
                    ]),
                    "overlay_capacity_bytes": int(runs[0]["scan_attention_overlay_capacity_bytes"]),
                    "overlay_required_bytes": int(runs[0]["scan_attention_overlay_required_bytes"]),
                    "physical_pass": physical,
                    "output_hash": next(iter(output_hashes)) if len(output_hashes) == 1 else None,
                    "cache_k_hash": next(iter(k_hashes)) if len(k_hashes) == 1 else None,
                    "cache_v_hash": next(iter(v_hashes)) if len(v_hashes) == 1 else None,
                    "reference_max_lsb": max(int(r["max_lsb"]) for r in runs),
                    "reference_cache_mismatches": max(
                        int(r["scan_cache_mismatches"]) for r in runs
                    ),
                }

    equivalence: dict[str, bool] = {}
    for length in LENGTHS:
        for repeat_count in REPEATS:
            valid = [
                cells[f"l{length}_r{repeat_count}_{mode}"]
                for mode in MODES
                if cells[f"l{length}_r{repeat_count}_{mode}"].get("available")
            ]
            equivalence[f"l{length}_r{repeat_count}"] = bool(valid) and all(
                item["output_hash"] == valid[0]["output_hash"]
                and item["cache_k_hash"] == valid[0]["cache_k_hash"]
                and item["cache_v_hash"] == valid[0]["cache_v_hash"]
                for item in valid
            )

    summary = {
        "experiment": "EXP-0161",
        "phase": "A_monolithic_delta_length_scaling",
        "cells": cells,
        "mode_equivalence": equivalence,
        "unavailable_runs": failures,
        "reference_note": (
            "EXP0147 independent CPU reference has the pre-existing <=2 LSB "
            "full-block and captured-token cache discrepancy; physical-mode "
            "equivalence is additionally checked by exact output/cache hashes"
        ),
    }
    (result_dir / "phase_a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0161 Phase A: monolithic delta-cache length scaling",
        "",
        "Each cell is the median of five device runs. Times are per complete layer-14 block.",
        "",
        "| L | Repeat | Mode | Host wall | Attention | K pack | V pack | Pipeline wait | Overlay required/capacity |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for length in LENGTHS:
        for repeat_count in REPEATS:
            for mode in MODES:
                cell = cells[f"l{length}_r{repeat_count}_{mode}"]
                if not cell.get("available"):
                    lines.append(
                        f"| {length} | {repeat_count} | {mode} | unavailable | unavailable | unavailable | unavailable | unavailable | capacity/finalize failure |"
                    )
                    continue
                lines.append(
                    f"| {length} | {repeat_count} | {mode} | "
                    f"{cell['host_us_per_block']:.3f} us | "
                    f"{cell['attention_us_per_block']:.3f} us | "
                    f"{cell['k_pack_us_per_block']:.3f} us | "
                    f"{cell['v_pack_us_per_block']:.3f} us | "
                    f"{cell['pipeline_wait_us_per_block']:.3f} us | "
                    f"{cell['overlay_required_bytes']}/{cell['overlay_capacity_bytes']} B |"
                )
    lines += ["", "## Gates", ""]
    for key, value in equivalence.items():
        lines.append(f"- {key} exact mode output/cache equivalence: {'PASS' if value else 'FAIL'}")
    lines += [
        "",
        "The inherited EXP-0147 software/captured reference discrepancy is reported separately and is not hidden by the mode-equivalence check.",
    ]
    (result_dir / "phase_a_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result_dir": str(result_dir),
        "cells": len(cells),
        "unavailable_runs": len(failures),
        "mode_equivalence": equivalence,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
