#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


STORAGES_AND_PLANS = (
    ("expanded_s8_control", "exp0005_full_bundle_control"),
    ("packed_w4u8", "exp0006_slots2_chunk32_control"),
    ("packed_w4u8", "slots3_chunk32"),
    ("packed_w4u8", "slots4_chunk32"),
    ("packed_w4u8", "slots2_chunk16"),
    ("packed_w4u8", "slots3_chunk16"),
)
PLAN_CONFIG = {
    "exp0005_full_bundle_control": (1, 2, 32),
    "exp0006_slots2_chunk32_control": (6, 2, 32),
    "slots3_chunk32": (6, 3, 32),
    "slots4_chunk32": (6, 4, 32),
    "slots2_chunk16": (6, 2, 16),
    "slots3_chunk16": (6, 3, 16),
}
PACKED_CONTROL = "exp0006_slots2_chunk32_control"
CANDIDATE_PLANS = (
    "slots3_chunk32",
    "slots4_chunk32",
    "slots2_chunk16",
    "slots3_chunk16",
)
PROJECTIONS = ("gate_up", "down")
PATTERNS = ("identity", "signed", "structured", "boundary")


def load_jsonl(path: pathlib.Path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def key(record):
    return (
        record["weight_storage"],
        record["physical_plan"],
        record["projection"],
        record["pattern"],
    )


def assert_runtime_valid(record, repeat):
    record_key = key(record)
    if record["experiment"] != "EXP-0007":
        raise SystemExit(f"wrong experiment for {record_key}")
    if record["repeat_count"] != repeat:
        raise SystemExit(f"wrong repeat for {record_key}")
    if record["rpc_result"] != 0 or record["dsp_status"] != 0:
        raise SystemExit(f"runtime failure for {record_key}")
    if record["mismatches"] != 0:
        raise SystemExit(f"output mismatch for {record_key}")
    if record["dma_status"] != 0 or record["sync_status"] != 0:
        raise SystemExit(f"DMA/sync failure for {record_key}")
    if record["dcvs_power_setup_status"] != 0:
        raise SystemExit(f"DCVS setup failure for {record_key}")
    if record["dcvs_power_reset_status"] != 0:
        raise SystemExit(f"DCVS reset failure for {record_key}")
    if record["hmx_execution_count"] != 12288 * repeat:
        raise SystemExit(f"wrong HMX execution count for {record_key}")

    workers, compressed_slots, chunk_tiles = PLAN_CONFIG[
        record["physical_plan"]
    ]
    if record["requested_hvx_workers"] != workers:
        raise SystemExit(f"wrong requested workers for {record_key}")
    if record["compressed_slot_count"] != compressed_slots:
        raise SystemExit(f"wrong compressed-slot count for {record_key}")
    if record["chunk_tiles"] != chunk_tiles:
        raise SystemExit(f"wrong chunk depth for {record_key}")

    if record["weight_storage"] == "packed_w4u8":
        if record["hvx_workers_created"] != workers:
            raise SystemExit(f"worker creation fallback for {record_key}")
        if record["hvx_workers_locked"] != workers:
            raise SystemExit(f"worker lock fallback for {record_key}")
        if record["hvx_hmx_overlap_observed"] != 1:
            raise SystemExit(f"no HVX/HMX overlap for {record_key}")
        if record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"no multi-HVX overlap for {record_key}")
        if record["chunk_expand_count"] == 0:
            raise SystemExit(f"no chunk expansion for {record_key}")


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return grouped


def median_metrics(records):
    fields = (
        "host_wall_ns",
        "pipeline_ticks",
        "weight_stage_ticks",
        "weight_expand_ticks",
        "hmx_compute_ticks",
        "hmx_ready_wait_ticks",
        "producer_slot_wait_ticks",
        "expanded_slot_wait_ticks",
        "hvx_max_active_workers",
    )
    return {
        f"{field}_median": statistics.median(
            record[field] for record in records
        )
        for field in fields
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    expected_keys = {
        (storage, plan, projection, pattern)
        for storage, plan in STORAGES_AND_PLANS
        for projection in PROJECTIONS
        for pattern in PATTERNS
    }

    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing = load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = group_records(correctness)
    timing_groups = group_records(timing)
    if set(correctness_groups) != expected_keys:
        raise SystemExit("correctness matrix has missing/unexpected keys")
    if set(timing_groups) != expected_keys:
        raise SystemExit("timing matrix has missing/unexpected keys")
    if any(len(records) != 1 for records in correctness_groups.values()):
        raise SystemExit("correctness matrix must have one record per key")
    if any(len(records) != 3 for records in timing_groups.values()):
        raise SystemExit("timing matrix must have three records per key")

    for record in correctness:
        assert_runtime_valid(record, 1)
    for record in timing:
        assert_runtime_valid(record, 10)

    comparable = (
        "reference_checksum",
        "reference_min",
        "reference_max",
        "expanded_carrier_checksum",
        "projection_m",
        "projection_k",
        "projection_n",
        "hmx_execution_count",
        "output_tile_count",
    )
    for grouped in (correctness_groups, timing_groups):
        for projection in PROJECTIONS:
            for pattern in PATTERNS:
                peers = [
                    record
                    for group_key, records in grouped.items()
                    if group_key[2:] == (projection, pattern)
                    for record in records
                ]
                reference = peers[0]
                for record in peers[1:]:
                    for field in comparable:
                        if record[field] != reference[field]:
                            raise SystemExit(
                                f"{projection}/{pattern}: variants differ at {field}"
                            )

    summary = {}
    local_performance_gate = True
    all_stretch_targets = True
    for projection in PROJECTIONS:
        plan_metrics = {}
        for storage, plan in STORAGES_AND_PLANS:
            records = [
                record
                for group_key, group in timing_groups.items()
                if group_key[0] == storage
                and group_key[1] == plan
                and group_key[2] == projection
                for record in group
            ]
            plan_metrics[f"{storage}:{plan}"] = median_metrics(records)

        packed_control = plan_metrics[
            f"packed_w4u8:{PACKED_CONTROL}"
        ]["host_wall_ns_median"]
        expanded_control = plan_metrics[
            "expanded_s8_control:exp0005_full_bundle_control"
        ]["host_wall_ns_median"]
        selected_candidate = min(
            CANDIDATE_PLANS,
            key=lambda plan: plan_metrics[f"packed_w4u8:{plan}"][
                "host_wall_ns_median"
            ],
        )
        candidate_wall = plan_metrics[
            f"packed_w4u8:{selected_candidate}"
        ]["host_wall_ns_median"]
        best_packed = min(
            (PACKED_CONTROL, *CANDIDATE_PLANS),
            key=lambda plan: plan_metrics[f"packed_w4u8:{plan}"][
                "host_wall_ns_median"
            ],
        )
        best_packed_wall = plan_metrics[f"packed_w4u8:{best_packed}"][
            "host_wall_ns_median"
        ]
        improves_parent = candidate_wall < packed_control
        beats_expanded = best_packed_wall < expanded_control
        local_performance_gate &= improves_parent
        all_stretch_targets &= beats_expanded
        summary[projection] = {
            "selected_candidate_plan": selected_candidate,
            "selected_packed_plan_including_control": best_packed,
            "packed_control_host_wall_ns_median": packed_control,
            "selected_candidate_host_wall_ns_median": candidate_wall,
            "candidate_improvement_percent":
                (packed_control - candidate_wall) * 100.0 / packed_control,
            "beats_exp0006_control": improves_parent,
            "expanded_s8_host_wall_ns_median": expanded_control,
            "best_packed_host_wall_ns_median": best_packed_wall,
            "packed_vs_expanded_improvement_percent":
                (expanded_control - best_packed_wall)
                * 100.0
                / expanded_control,
            "beats_expanded_s8": beats_expanded,
            "metrics": plan_metrics,
        }

    packed_sample = correctness_groups[
        ("packed_w4u8", PACKED_CONTROL, "gate_up", "identity")
    ][0]
    expanded_sample = correctness_groups[
        (
            "expanded_s8_control",
            "exp0005_full_bundle_control",
            "gate_up",
            "identity",
        )
    ][0]
    if not (
        packed_sample["stored_weight_bytes_per_repeat"]
        < expanded_sample["stored_weight_bytes_per_repeat"]
    ):
        raise SystemExit("packed W4 did not reduce Native DDR weight bytes")

    print(
        json.dumps(
            {
                "experiment": "EXP-0007",
                "records": len(correctness) + len(timing),
                "correctness_records": len(correctness),
                "timing_records": len(timing),
                "exact_reference": True,
                "all_variants_equivalent": True,
                "worker_fallback": False,
                "dcvs_performance_vote": True,
                "hvx_hmx_overlap": True,
                "local_performance_gate": local_performance_gate,
                "beats_expanded_s8_both_shapes": all_stretch_targets,
                "summary": summary,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
