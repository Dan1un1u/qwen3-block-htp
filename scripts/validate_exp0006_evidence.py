#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


STORAGES_AND_PLANS = (
    ("expanded_s8_control", "exp0005_full_bundle_control"),
    ("packed_w4u8", "exp0005_full_bundle_control"),
    ("packed_w4u8", "chunked_hvx1"),
    ("packed_w4u8", "chunked_hvx2"),
    ("packed_w4u8", "chunked_hvx4"),
    ("packed_w4u8", "chunked_hvx6"),
)
PROJECTIONS = ("gate_up", "down")
PATTERNS = ("identity", "signed", "structured", "boundary")
CHUNKED_PLANS = (
    "chunked_hvx1",
    "chunked_hvx2",
    "chunked_hvx4",
    "chunked_hvx6",
)


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
    if record["experiment"] != "EXP-0006":
        raise SystemExit(f"wrong experiment for {record_key}")
    if record["repeat_count"] != repeat:
        raise SystemExit(f"wrong repeat for {record_key}")
    if record["rpc_result"] != 0 or record["dsp_status"] != 0:
        raise SystemExit(f"runtime failure for {record_key}")
    if record["mismatches"] != 0:
        raise SystemExit(f"output mismatch for {record_key}")
    if record["dma_status"] != 0 or record["sync_status"] != 0:
        raise SystemExit(f"DMA/sync failure for {record_key}")
    if record["hmx_execution_count"] != 12288 * repeat:
        raise SystemExit(f"wrong HMX execution count for {record_key}")
    if record["physical_plan"].startswith("chunked_hvx"):
        workers = int(record["physical_plan"].removeprefix("chunked_hvx"))
        if record["requested_hvx_workers"] != workers:
            raise SystemExit(f"wrong requested workers for {record_key}")
        if record["hvx_workers_created"] != workers:
            raise SystemExit(f"worker creation fallback for {record_key}")
        if record["hvx_workers_locked"] != workers:
            raise SystemExit(f"worker lock fallback for {record_key}")
        if record["hvx_hmx_overlap_observed"] != 1:
            raise SystemExit(f"no HVX/HMX overlap for {record_key}")
        if workers > 1 and record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"no multi-HVX overlap for {record_key}")
        if record["chunk_expand_count"] == 0:
            raise SystemExit(f"no chunk expansion for {record_key}")


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return grouped


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
        "hmx_stream_count",
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
            label = f"{storage}:{plan}"
            plan_metrics[label] = {
                "host_wall_ns_median": statistics.median(
                    record["host_wall_ns"] for record in records
                ),
                "pipeline_ticks_median": statistics.median(
                    record["pipeline_ticks"] for record in records
                ),
                "weight_expand_ticks_sum_median": statistics.median(
                    record["weight_expand_ticks"] for record in records
                ),
                "hmx_ready_wait_ticks_median": statistics.median(
                    record["hmx_ready_wait_ticks"] for record in records
                ),
            }

        packed_control = plan_metrics[
            "packed_w4u8:exp0005_full_bundle_control"
        ]["host_wall_ns_median"]
        selected_plan = min(
            CHUNKED_PLANS,
            key=lambda plan: plan_metrics[f"packed_w4u8:{plan}"][
                "host_wall_ns_median"
            ],
        )
        selected_wall = plan_metrics[f"packed_w4u8:{selected_plan}"][
            "host_wall_ns_median"
        ]
        improved = selected_wall < packed_control
        local_performance_gate &= improved
        summary[projection] = {
            "selected_chunked_plan": selected_plan,
            "packed_control_host_wall_ns_median": packed_control,
            "selected_host_wall_ns_median": selected_wall,
            "host_wall_improvement_percent":
                (packed_control - selected_wall) * 100.0 / packed_control,
            "beats_packed_control": improved,
            "metrics": plan_metrics,
        }

    packed_sample = correctness_groups[
        ("packed_w4u8", "chunked_hvx1", "gate_up", "identity")
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
    if not local_performance_gate:
        raise SystemExit("selected chunked plan did not beat packed control")

    print(
        json.dumps(
            {
                "experiment": "EXP-0006",
                "records": len(correctness) + len(timing),
                "correctness_records": len(correctness),
                "timing_records": len(timing),
                "exact_reference": True,
                "all_variants_equivalent": True,
                "worker_fallback": False,
                "hvx_hmx_overlap": True,
                "local_performance_gate": True,
                "summary": summary,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
