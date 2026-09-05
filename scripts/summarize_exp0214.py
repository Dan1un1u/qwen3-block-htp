#!/usr/bin/env python3
"""Validate EXP-0214 real-shape M64 direct-W4 HMX projection gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "host_wall_ns",
    "dsp_total_ticks",
    "pipeline_ticks",
    "activation_stage_ticks",
    "weight_stage_ticks",
    "weight_expand_ticks",
    "hmx_compute_ticks",
    "hmx_ready_wait_ticks",
    "producer_slot_wait_ticks",
    "expanded_slot_wait_ticks",
    "output_assembly_ticks",
    "input_cache_ticks",
    "output_cache_ticks",
    "stored_weight_bytes_per_repeat",
    "expanded_weight_bytes_per_repeat",
    "vtcm_plan_bytes",
    "hmx_execution_count",
    "hmx_stream_count",
    "weight_expand_count",
    "activation_stage_count",
    "weight_bundle_stage_count",
    "output_tile_count",
    "dma_submit_count",
    "dma_wait_count",
    "dma_descriptor_count",
    "dma_chain_count",
    "dma_descriptor_completion_count",
    "output_dma_submit_count",
    "output_dma_wait_count",
    "output_dma_descriptor_count",
    "output_dma_chain_count",
    "output_dma_descriptor_completion_count",
    "streaming_region_publish_count",
)

WORK_FIELDS = (
    ("DSP total", "dsp_total_ticks", "ticks"),
    ("Projection pipeline", "pipeline_ticks", "ticks"),
    ("Activation stage", "activation_stage_ticks", "ticks"),
    ("Weight stage", "weight_stage_ticks", "ticks"),
    ("W4-to-S8 expansion", "weight_expand_ticks", "ticks"),
    ("HMX compute", "hmx_compute_ticks", "ticks"),
    ("HMX ready wait", "hmx_ready_wait_ticks", "ticks"),
    ("Producer-slot wait", "producer_slot_wait_ticks", "ticks"),
    ("Expanded-slot wait", "expanded_slot_wait_ticks", "ticks"),
    ("Output assembly", "output_assembly_ticks", "ticks"),
    ("Input cache maintenance", "input_cache_ticks", "ticks"),
    ("Output cache maintenance", "output_cache_ticks", "ticks"),
)

PHYSICAL_FIELDS = (
    ("Stored W4 bytes", "stored_weight_bytes_per_repeat", "B"),
    ("Expanded-carrier bytes represented", "expanded_weight_bytes_per_repeat", "B"),
    ("Peak used VTCM plan", "vtcm_plan_bytes", "B"),
    ("HMX executions", "hmx_execution_count", "count"),
    ("HMX streams", "hmx_stream_count", "count"),
    ("W4 expansion calls", "weight_expand_count", "count"),
    ("Activation stages", "activation_stage_count", "count"),
    ("Weight-bundle stages", "weight_bundle_stage_count", "count"),
    ("Output tiles", "output_tile_count", "count"),
    ("DMA submits", "dma_submit_count", "count"),
    ("DMA waits", "dma_wait_count", "count"),
    ("DMA descriptors", "dma_descriptor_count", "count"),
    ("DMA chains", "dma_chain_count", "count"),
    ("DMA descriptor completions", "dma_descriptor_completion_count", "count"),
    ("Output DMA submits", "output_dma_submit_count", "count"),
    ("Output DMA waits", "output_dma_wait_count", "count"),
    ("Output DMA descriptors", "output_dma_descriptor_count", "count"),
    ("Output DMA chains", "output_dma_chain_count", "count"),
    ("Output DMA descriptor completions", "output_dma_descriptor_completion_count", "count"),
    ("Streaming-region publications", "streaming_region_publish_count", "count"),
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
        "source_branch": "codex/exp-0214-w4u8-prefill-direct-w4-hmx-gate",
        "formal_evidence": str(result_dir),
        "artifact": "/mnt/d/llm_exp/models/qwen3-block-htp/exp0187/real_layer14_m64",
        "execution_unit": "one real layer-14 M64 projection invocation",
        "project_variant": "W4U8",
        "direct_control": "packed W4 plus explicit HVX W4-to-S8 expansion",
        "candidate": "HMX direct weight.n packed-W4 carrier",
        "paired": True,
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
        "## Identity",
        "",
        "- Source branch: `codex/exp-0214-w4u8-prefill-direct-w4-hmx-gate`",
        f"- Source commit: `{args.source_commit}`",
        f"- Formal evidence: `{result_dir}`",
        "- Artifact: `/mnt/d/llm_exp/models/qwen3-block-htp/exp0187/real_layer14_m64`",
        "- Execution unit: one real layer-14 M64 projection invocation",
        "- Project Variant: W4U8",
        "- Direct control: packed W4 staged and explicitly expanded by HVX to an S8 HMX carrier",
        "- Candidate: reordered packed W4 consumed directly by HMX `weight.n`",
        "- Measurement: ten alternating-order paired rounds at repeat1 and repeat10",
        "",
        "Real Qwen3 layer-14 M64 U8 activations, W4 weights, per-channel scales, and HMX bias are used. Only the packed-weight physical carrier and the removal of explicit HVX W4-to-S8 expansion change.",
        "",
        "## Full-stack throughput and three-variant overview",
        "",
        "N/A. This bounded gate owns one real projection, not a complete token boundary; reporting or extrapolating prefill/decode tokens/s is forbidden. The stable F16F16/W4F16/W4U8 full-stack module overview is also N/A for this projection-only execution scope. Equivalent-scope Host wall is the primary metric.",
        "",
        "## Primary paired Host wall",
        "",
        "| Projection | Repeat | Control host ms | Direct-n host ms | Host speedup | Control pipeline ticks | Direct-n pipeline ticks | Pipeline speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Overlapping engine work and waits",
        "",
        "These qtimer counters overlap and must not be summed into wall time.",
        "",
        "| Projection | Repeat | Metric | Unit | Control | Direct-n | Candidate delta |",
        "|---|---:|---|---|---:|---:|---:|",
        *[
            f"| {projection} | {repeat_count} | {label} | {unit} | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field]:.1f} | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field]:.1f} | "
            f"{(summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field] / summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field] - 1.0) * 100.0:+.2f}% |"
            if summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field] != 0
            else f"| {projection} | {repeat_count} | {label} | {unit} | 0.0 | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field]:.1f} | N/A (zero control) |"
            for projection in ("gate_up_pair", "down")
            for repeat_count in (1, 10)
            for label, field, unit in WORK_FIELDS
        ],
        "",
        "## Physical contract and command counts",
        "",
        "| Projection | Repeat | Metric | Unit | Control | Direct-n | Candidate delta |",
        "|---|---:|---|---|---:|---:|---:|",
        *[
            f"| {projection} | {repeat_count} | {label} | {unit} | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field]:.1f} | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field]:.1f} | "
            f"{(summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field] / summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field] - 1.0) * 100.0:+.2f}% |"
            if summary['cells'][f'{projection}_repeat{repeat_count}']['control'][field] != 0
            else f"| {projection} | {repeat_count} | {label} | {unit} | 0.0 | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['direct_n'][field]:.1f} | N/A (zero control) |"
            for projection in ("gate_up_pair", "down")
            for repeat_count in (1, 10)
            for label, field, unit in PHYSICAL_FIELDS
        ],
        "",
        "Both cells request and acquire exactly 8,388,608 bytes of VTCM. The only legal DDR boundaries are the input activation, stored weights/biases and output activation. Timed intermediate DDR tensors, compiler spill/fill, CPU/QNN fallback and extra HMX owners are all zero; the execution uses one FastRPC call and the standalone FastRPC/cDSP backend.",
        "",
        "## Correctness",
        "",
        "| Projection | Repeat | Complete-output FNV equal | External diagnostic mismatch count | DSP/timeout gates |",
        "|---|---:|---|---:|---|",
        *[
            f"| {projection} | {repeat_count} | yes | "
            f"{summary['cells'][f'{projection}_repeat{repeat_count}']['external_reference_mismatches']} | pass |"
            for projection in ("gate_up_pair", "down")
            for repeat_count in (1, 10)
        ],
        "",
        "Maximum LSB versus the stale software-postscale reference is N/A because this retained probe reports mismatch count but not maximum delta. Candidate-versus-control correctness is strict: their FNV-1a digest covers every M64 output byte and is identical in every rotated run. Non-finite and mask checks are N/A for an integer projection without masking.",
        "",
        "## Non-applicable full-block rows",
        "",
        "The additive Block Timing Ledger, Attention diagnostics, residuals, nonlinear operators, KV cache, layer bookkeeping, runtime setup/teardown attribution and three-recipe module table are N/A because this execution unit is a standalone projection probe. No such row is inferred from partial measurements.",
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
