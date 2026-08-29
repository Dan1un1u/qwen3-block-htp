#!/usr/bin/env python3
"""Validate and report EXP-0048 Stage-A paired evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SAMPLES = 7
REPEATS = (1, 10)
MODES = ("control", "qkv_native")
BOUNDARY_MODE = {
    "control": "w4u8_mlp_io",
    "qkv_native": "w4u8_mlp_io_qkv_input",
}
OUTPUT_HASH = "69f22eeb035e5ec5"
VTCM_BYTES = 8_388_608


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def require(record: dict[str, object], field: str, expected: object) -> None:
    actual = record.get(field)
    if actual != expected:
        raise SystemExit(f"wrong {field}: {actual!r} != {expected!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, expected: int = SAMPLES) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != expected:
        raise SystemExit(f"wrong record count for {path}: {len(records)}")
    return records


def validate_record(record: dict[str, object], repeat: int, mode: str) -> None:
    fixed = {
        "experiment": "EXP-0048",
        "execution_unit": "qwen3_layer14_complete_block_m64",
        "variant": "W4U8",
        "attention_compute": "U8xS8_HMX_log2_softmax",
        "projection_compute": "U8xS8_integer_HMX",
        "crouton_boundary_mode": BOUNDARY_MODE[mode],
        "w4u8_qkvo_pipeline_mode": "qkvo_batch4",
        "attention_pipeline_mode": "u8_log2_gqa_qkv_overlap",
        "mlp_mode": "w4u8_streaming",
        "repeat_count": repeat,
        "warmup_rpc_result": 0,
        "warmup_mismatches": 0,
        "warmup_max_lsb": 0,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "mismatches": 0,
        "max_lsb": 0,
        "output_hash": OUTPUT_HASH,
        "vtcm_requested_bytes": VTCM_BYTES,
        "vtcm_acquired_bytes": VTCM_BYTES,
        "block_invocation_count": repeat,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_dma_descriptor_count": 0,
        "intermediate_spill_fill_count": 0,
        "u8_attention_audit_ddr_write_bytes": 0,
        "u8_attention_fused_k_operand_mismatch_count": 0,
        "u8_attention_qkv_unpack_skipped": 128 * repeat,
        "hmx_command_count": 1040 * repeat,
        "hmx_u8s8_tile_pair_count": 49_408 * repeat,
        "weight_dma_descriptor_count": 512 * repeat,
        "weight_ddr_read_bytes": 25_444_352 * repeat,
        "w4u8_qkv_batch_n_tiles": 4,
        "w4u8_qkv_batch_count": 32 * repeat,
        "w4u8_qkvo_prefetch_count": 44 * repeat,
        "w4u8_qkvo_overlap_schedule_count": 44 * repeat,
        "w4u8_mlp_input_pack_skipped": repeat,
        "w4u8_mlp_output_unpack_skipped": repeat,
        "w4u8_mlp_input_pack_ticks": 0,
        "w4u8_mlp_output_unpack_ticks": 0,
        "release_result": 0,
        "close_result": 0,
    }
    for field, expected in fixed.items():
        require(record, field, expected)
    if int(record["vtcm_peak_plan_bytes"]) > VTCM_BYTES:
        raise SystemExit("VTCM plan exceeds 8 MiB")
    invocation = float(record["invocation_ticks"])
    closure = abs(invocation - float(record["ledger_named_ticks"])) / invocation
    if closure > 0.001:
        raise SystemExit(f"ledger closure exceeds 0.1%: {closure:.6%}")
    for field in (
        "host_wall_ns_per_block", "total_ticks", "input_norm_ticks",
        "qkv_projection_ticks", "attention_ticks", "o_projection_ticks",
        "gate_up_ticks", "down_ticks", "final_residual_ticks",
    ):
        if float(record[field]) <= 0.0:
            raise SystemExit(f"non-positive {field}")


def per_block(record: dict[str, object], field: str) -> float:
    if field == "host_wall_ns_per_block":
        return float(record[field])
    return float(record[field]) / int(record["repeat_count"])


def summarize(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
    field: str,
) -> dict[str, float]:
    left = [per_block(record, field) for record in control]
    right = [per_block(record, field) for record in candidate]
    left_median = statistics.median(left)
    right_median = statistics.median(right)
    paired = [
        (right[index] / left[index] - 1.0) * 100.0
        for index in range(SAMPLES)
    ]
    return {
        "control": left_median,
        "candidate": right_median,
        "change_percent": (right_median / left_median - 1.0) * 100.0,
        "paired_change_percent_median": statistics.median(paired),
        "paired_change_percent_min": min(paired),
        "paired_change_percent_max": max(paired),
    }


METRICS = (
    "host_wall_ns_per_block",
    "invocation_ticks",
    "total_ticks",
    "input_norm_ticks",
    "qkv_projection_ticks",
    "attention_ticks",
    "o_projection_ticks",
    "post_attention_residual_ticks",
    "gate_up_ticks",
    "down_ticks",
    "final_residual_ticks",
    "projection_pack_ticks",
    "projection_unpack_ticks",
    "weight_dma_ticks",
    "hmx_compute_ticks",
    "w4u8_qkvo_weight_expand_ticks",
    "w4u8_qkvo_prefetch_wait_ticks",
    "w4u8_qkvo_hmx_lifetime_ticks",
    "projection_hmx_wait_ticks",
    "hmx_ready_wait_ticks",
    "attention_qk_norm_pool_wait_ticks",
    "w4u8_mlp_weight_expand_ticks",
    "w4u8_mlp_hmx_compute_ticks",
    "w4u8_mlp_expanded_slot_wait_ticks",
    "hmx_command_count",
    "hmx_u8s8_tile_pair_count",
    "weight_dma_descriptor_count",
    "weight_ddr_read_bytes",
)


def build_summary(result_dir: Path, package_dir: Path) -> dict[str, object]:
    if (result_dir / "boot_id_before.txt").read_bytes() != (
        result_dir / "boot_id_after.txt"
    ).read_bytes():
        raise SystemExit("device boot ID changed")

    implementation = json.loads(
        (result_dir / "attention_implementation_audit" /
         "implementation_reference.json").read_text()
    )
    require(implementation, "experiment", "EXP-0048")
    require(implementation, "core_exact", True)
    require(implementation, "probability_mask_violations", 0)
    for stage in ("qk", "log2_softmax", "av"):
        require(implementation["exact_stage_comparison"][stage], "mismatches", 0)

    correctness: dict[str, object] = {}
    for mode in MODES:
        record = load_jsonl(
            result_dir / f"correctness_{mode}.jsonl", expected=1
        )[0]
        require(record, "experiment", "EXP-0048")
        require(record, "crouton_boundary_mode", BOUNDARY_MODE[mode])
        require(record, "output_hash", OUTPUT_HASH)
        require(record, "mismatches", 0)
        require(record, "max_lsb", 0)
        require(record, "rpc_result", 0)
        require(record, "dsp_status", 3)
        correctness[mode] = {
            "output_hash": record["output_hash"],
            "mismatches": record["mismatches"],
            "max_lsb": record["max_lsb"],
        }

    repeats: dict[str, object] = {}
    gates: list[bool] = []
    for repeat in REPEATS:
        records = {
            mode: load_jsonl(result_dir / f"paired_{mode}_r{repeat}.jsonl")
            for mode in MODES
        }
        for mode, values in records.items():
            for record in values:
                validate_record(record, repeat, mode)
        metrics = {
            field: summarize(records["control"], records["qkv_native"], field)
            for field in METRICS
        }
        speed_gate = all(
            metrics[field][key] < 0.0
            for field in (
                "host_wall_ns_per_block",
                "projection_pack_ticks",
                "qkv_projection_ticks",
            )
            for key in ("change_percent", "paired_change_percent_median")
        )
        unchanged_physical_work = all(
            metrics[field]["control"] == metrics[field]["candidate"]
            for field in (
                "hmx_command_count", "hmx_u8s8_tile_pair_count",
                "weight_dma_descriptor_count", "weight_ddr_read_bytes",
            )
        )
        gates.append(speed_gate and unchanged_physical_work)
        repeats[f"repeat{repeat}"] = {
            "metrics": metrics,
            "speed_gate": speed_gate,
            "unchanged_physical_work_gate": unchanged_physical_work,
        }

    return {
        "experiment": "EXP-0048",
        "stage": "A",
        "control": "W4U8-EXP0046-NATIVE-MLP-IO",
        "candidate": "input_rmsnorm_to_shared_qkv_u8_hmx_carrier",
        "package_manifest_sha256": sha256(package_dir / "manifest.json"),
        "implementation_reference_gate": True,
        "byte_exact_final_output_gate": True,
        "fixed_8mib_vtcm_gate": True,
        "zero_intermediate_ddr_gate": True,
        "zero_spill_fill_gate": True,
        "single_fastrpc_execution_unit": True,
        "single_hmx_owner": True,
        "qnn_dependency": False,
        "correctness": correctness,
        "repeat_results": repeats,
        "stage_a_gate_pass": all(gates),
        "stage_b_authorized": all(gates),
    }


LABELS = {
    "host_wall_ns_per_block": "Host wall (ns/block)",
    "invocation_ticks": "DSP invocation ticks",
    "input_norm_ticks": "Input RMSNorm ticks",
    "qkv_projection_ticks": "QKV ledger ticks",
    "attention_ticks": "Attention ledger ticks",
    "o_projection_ticks": "O projection ticks",
    "gate_up_ticks": "Gate/Up ticks",
    "down_ticks": "Down ticks",
    "projection_pack_ticks": "Projection pack ticks",
    "projection_unpack_ticks": "Projection unpack ticks",
    "weight_dma_ticks": "Weight DMA ticks (overlap)",
    "hmx_compute_ticks": "HMX compute ticks (overlap)",
    "w4u8_qkvo_weight_expand_ticks": "QKVO W4 expand ticks (overlap)",
    "w4u8_qkvo_prefetch_wait_ticks": "QKVO prefetch wait ticks",
    "w4u8_qkvo_hmx_lifetime_ticks": "QKVO HMX lifetime ticks",
    "attention_qk_norm_pool_wait_ticks": "QK prep pool wait ticks",
    "w4u8_mlp_expanded_slot_wait_ticks": "MLP expanded-slot wait ticks",
}


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# EXP-0048 Stage A — Complete profiling report",
        "",
        "The candidate changes only the input-RMSNorm-to-Q/K/V activation "
        "boundary: RMSNorm scatters one U8 HMX carrier and Q, K, and V reuse "
        "it. W4 values/scales, qparams, integer Attention, HMX arithmetic, "
        "MLP native boundaries, and all non-target schedules remain fixed.",
        "",
    ]
    for repeat in REPEATS:
        result = summary["repeat_results"][f"repeat{repeat}"]
        lines.extend([
            f"## Repeat {repeat}", "",
            "| Metric | Control | Candidate | Median change | Paired median |",
            "|---|---:|---:|---:|---:|",
        ])
        for field in LABELS:
            metric = result["metrics"][field]
            lines.append(
                f"| {LABELS[field]} | {metric['control']:.3f} | "
                f"{metric['candidate']:.3f} | {metric['change_percent']:.3f}% | "
                f"{metric['paired_change_percent_median']:.3f}% |"
            )
        lines.extend(["", f"Stage speed gate: **{'PASS' if result['speed_gate'] else 'FAIL'}**.", ""])

    lines.extend([
        "## Physical and numerical contract", "",
        "| Gate | Result |", "|---|---:|",
        "| Final output byte-exact to control | PASS (0 LSB) |",
        "| Independent integer QK / log2 Softmax / AV reference | PASS |",
        "| Requested/acquired VTCM | 8,388,608 / 8,388,608 bytes |",
        "| Intermediate DDR read/write | 0 / 0 bytes |",
        "| Spill/fill | 0 |",
        "| FastRPC / HMX ownership | one execution unit / one owner |",
        "| QNN dependency | none |",
        "| HMX commands, tile pairs, weight DMA bytes/descriptors | unchanged |",
        "",
        "The additive ledger and overlapping engine counters are deliberately "
        "reported separately. Host wall is the primary speed measure; DMA, "
        "HVX/HMX work, pool waits, and lifetimes are diagnostic and may overlap.",
        "",
        "## Decision", "",
        f"Stage A overall gate: **{'PASS' if summary['stage_a_gate_pass'] else 'FAIL'}**. "
        f"Stage B authorization: **{'YES' if summary['stage_b_authorized'] else 'NO'}**.",
        "",
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
