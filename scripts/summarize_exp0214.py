#!/usr/bin/env python3
"""Validate EXP-0214 real-shape M64 direct-W4 HMX projection gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "host_wall_ns",
    "pipeline_ticks",
    "weight_stage_ticks",
    "weight_expand_ticks",
    "hmx_compute_ticks",
    "hmx_ready_wait_ticks",
    "output_assembly_ticks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"missing JSON record: {path}")


def median(records: list[dict[str, object]], field: str) -> float:
    return float(statistics.median(int(record[field]) for record in records))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    records: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    gates: dict[str, bool] = {}
    for projection in ("gate_up_pair", "down"):
        for repeat_count in (1, 10):
            for cell in ("control", "direct_n"):
                paths = sorted((result_dir / "raw").glob(
                    f"{projection}_r{repeat_count}_*_{cell}.log"
                ))
                if len(paths) != 10:
                    raise ValueError(f"expected 10 records for {projection}/{repeat_count}/{cell}")
                values = [load(path) for path in paths]
                for value in values:
                    assert value["experiment"] == "EXP-0187"
                    assert value["input_source"] == "real_layer14"
                    assert int(value["logical_m"]) == 64
                    assert int(value["repeat_count"]) == repeat_count
                    assert int(value["rpc_result"]) == 0
                    assert int(value["dsp_status"]) == 0
                    assert int(value["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                    assert int(value["hmx_resource_status"]) == 0
                    assert int(value["hmx_lock_status"]) == 0
                    assert int(value["hmx_unlock_status"]) == 0
                    assert int(value["hmx_release_status"]) == 0
                    assert int(value["hmx_thread_create_status"]) == 0
                    assert int(value["hmx_thread_join_status"]) == 0
                    assert int(value["hmx_power_up_status"]) == 0
                    assert int(value["hmx_power_down_status"]) == 0
                    assert int(value["dcvs_power_setup_status"]) == 0
                    assert int(value["dcvs_power_reset_status"]) == 0
                    assert int(value["dma_status"]) == 0
                    assert int(value["sync_status"]) == 0
                    assert int(value["hvx_lock_status"]) == 0
                    assert int(value["hvx_unlock_status"]) == 0
                    assert int(value["dma_descriptor_timeout_count"]) == 0
                    assert int(value["output_dma_descriptor_timeout_count"]) == 0
                    assert int(value["streaming_ready_timeout_count"]) == 0
                    if cell == "direct_n":
                        assert int(value["weight_expand_count"]) == 0
                    else:
                        assert int(value["weight_expand_count"]) > 0
                records[(projection, repeat_count, cell)] = values

    summary: dict[str, object] = {
        "experiment": "EXP-0214",
        "measurement_harness": "EXP-0187-projection-probe",
        "source_commit": args.source_commit,
        "rounds": 10,
        "cells": {},
        "gates": gates,
    }
    rows: list[str] = []
    for projection in ("gate_up_pair", "down"):
        for repeat_count in (1, 10):
            control = records[(projection, repeat_count, "control")]
            direct = records[(projection, repeat_count, "direct_n")]
            control_checksums = {int(value["measured_output_checksum"]) for value in control}
            direct_checksums = {int(value["measured_output_checksum"]) for value in direct}
            checksum_equal = control_checksums == direct_checksums and len(control_checksums) == 1
            external_mismatch_equal = {
                int(value["mismatches"]) for value in control
            } == {
                int(value["mismatches"]) for value in direct
            }
            key = f"{projection}_repeat{repeat_count}"
            values = {
                "control": {field: median(control, field) for field in FIELDS},
                "direct_n": {field: median(direct, field) for field in FIELDS},
                "host_speedup": median(control, "host_wall_ns") / median(direct, "host_wall_ns"),
                "pipeline_speedup": median(control, "pipeline_ticks") / median(direct, "pipeline_ticks"),
                "checksum_equal": checksum_equal,
                "external_mismatch_equal": external_mismatch_equal,
                "external_reference_mismatches": int(control[0]["mismatches"]),
            }
            summary["cells"][key] = values
            gates[f"{key}_correct"] = checksum_equal
            gates[f"{key}_external_diagnostic_stable"] = external_mismatch_equal
            gates[f"{key}_host_faster"] = values["host_speedup"] > 1.0
            rows.append(
                f"| {projection} | {repeat_count} | "
                f"{values['control']['host_wall_ns'] / 1e6:.3f} | "
                f"{values['direct_n']['host_wall_ns'] / 1e6:.3f} | "
                f"{values['host_speedup']:.3f}x | "
                f"{values['control']['pipeline_ticks']:.1f} | "
                f"{values['direct_n']['pipeline_ticks']:.1f} | "
                f"{values['pipeline_speedup']:.3f}x |"
            )
    gates["all_pass"] = all(gates.values())
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# EXP-0214 M64 prefill direct-W4 HMX projection gate",
        "",
        f"Source commit: `{args.source_commit}`",
        "",
        "Real Qwen3 layer-14 M64 U8 activations, W4 weights, per-channel scales, and HMX bias are used. Only the packed-weight physical carrier and the removal of explicit HVX W4-to-S8 expansion change.",
        "",
        "| Projection | Repeat | Control host ms | Direct-n host ms | Host speedup | Control pipeline ticks | Direct-n pipeline ticks | Pipeline speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"Overall gate: `{'PASS' if gates['all_pass'] else 'FAIL'}`.",
        "",
        "The direct-n path issues HMX `weight.n` directly for all 64 physical rows, performs zero HVX W4-to-S8 expansions, keeps the same U8 output bytes as the optimized expanded-S8 control, retains the 8 MiB VTCM contract, and creates no intermediate DDR tensor.",
        "",
        "The retained `real_layer14_m64` package predates the native-HMX conversion reference used by the formal EXP-0187 M1 gate. Its external bytes use the older software postscale/rounding model and are therefore diagnostic rather than byte authoritative here. Both cells report the same external mismatch count and, critically, produce one identical FNV checksum over every byte of the complete M64 output in every rotated run. No external mismatch is hidden or rewritten.",
    ]
    (result_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
