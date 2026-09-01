#!/usr/bin/env python3
"""Summarize EXP-0147 scan JSON ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--label", default="shape/KV")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    objects = [line for line in lines if line.startswith("{") and line.endswith("}")]
    if len(objects) != 1:
        raise ValueError(f"{path}: expected one JSON object, got {len(objects)}")
    return json.loads(objects[0])


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for repeat10_path in sorted(args.input.glob("*_repeat10.json")):
        cell = repeat10_path.name.removesuffix("_repeat10.json")
        repeat1 = load_json(args.input / f"{cell}_repeat1.json")
        repeat10 = load_json(repeat10_path)
        logical_m = int(repeat10["logical_m"])
        block_ns = float(repeat10["host_wall_ns_per_block"])
        row = {
            "cell": cell,
            "mode": repeat10["scan_mode"],
            "logical_m": logical_m,
            "initial_kv_length": int(repeat10["initial_kv_length"]),
            "padded_kv_length": int(repeat10["scan_padded_kv_length"]),
            "repeat1_wall_us": float(repeat1["host_wall_ns_per_block"]) / 1000.0,
            "repeat10_wall_us": block_ns / 1000.0,
            "useful_tokens_per_s": logical_m * 1e9 / block_ns,
            "attention_us": int(repeat10["attention_ticks"]) / 19.2 / 10.0,
            "dynamic_attention_us": int(repeat10["scan_dynamic_attention_ticks"]) / 19.2 / 10.0,
            "cache_read_bytes_per_block": int(repeat10["scan_cache_ddr_read_bytes"]) / 10.0,
            "cache_write_bytes_per_block": int(repeat10["scan_cache_ddr_write_bytes"]) / 10.0,
            "vtcm_peak_bytes": int(repeat10["vtcm_peak_plan_bytes"]),
            "overlay_required_bytes": int(repeat10["scan_attention_overlay_required_bytes"]),
            "output_mismatches": int(repeat10["mismatches"]),
            "cache_mismatches": int(repeat10["scan_cache_mismatches"]),
            "intermediate_ddr_read_bytes": int(repeat10["intermediate_ddr_read_bytes"]),
            "intermediate_ddr_write_bytes": int(repeat10["intermediate_ddr_write_bytes"]),
            "spill_fill": int(repeat10["intermediate_spill_fill_count"]),
        }
        rows.append(row)

    args.output.write_text(
        json.dumps({"experiment": "EXP-0147", "cells": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# EXP-0147 {args.label} scan",
        "",
        "| Cell | repeat1 (us) | repeat10/block (us) | useful tok/s | Attention (us) | KV read/block | KV write/block | zero-LSB | zero intermediate DDR |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {cell} | {repeat1_wall_us:.3f} | {repeat10_wall_us:.3f} | "
            "{useful_tokens_per_s:.3f} | {attention_us:.3f} | "
            "{cache_read_bytes_per_block:.0f} B | {cache_write_bytes_per_block:.0f} B | "
            "{correct} | {zero_ddr} |".format(
                **row,
                correct="yes" if row["output_mismatches"] == 0 and row["cache_mismatches"] == 0 else "no",
                zero_ddr="yes" if row["intermediate_ddr_read_bytes"] == 0 and row["intermediate_ddr_write_bytes"] == 0 and row["spill_fill"] == 0 else "no",
            )
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.markdown.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
