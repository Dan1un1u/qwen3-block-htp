#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


PATTERNS = ("identity", "signed", "structured", "boundary")
SHAPES = ("gate_up_pair", "down")
PLANS = {
    "gate_up_pair": (
        "expanded_s8_dma_chain2",
        "slots8e7_chunk64_dma_chain4",
        "stream32_gate_hvx2",
        "stream32_gate_hvx3",
        "stream32_gate_hvx4",
        "stream32_gate_hvx6",
        "stream32_gate_hvx6_cap2",
    ),
    "down": (
        "exp0005_full_bundle_control",
        "slots4_chunk96_dma_batch2",
        "stream32_down_hvx2",
        "stream32_down_hvx3",
        "stream32_down_hvx4",
        "stream32_down_hvx6",
        "stream32_down_hvx6_cap2",
    ),
}
S8_PLAN = {
    "gate_up_pair": "expanded_s8_dma_chain2",
    "down": "exp0005_full_bundle_control",
}
PARENT_PLAN = {
    "gate_up_pair": "slots8e7_chunk64_dma_chain4",
    "down": "slots4_chunk96_dma_batch2",
}
STREAMING_PLANS = {
    shape: tuple(plan for plan in plans if plan.startswith("stream32_"))
    for shape, plans in PLANS.items()
}


def load_jsonl(path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def key(record):
    return record["projection"], record["physical_plan"], record["pattern"]


def expected_keys():
    return {
        (shape, plan, pattern)
        for shape in SHAPES
        for plan in PLANS[shape]
        for pattern in PATTERNS
    }


def is_streaming(plan):
    return plan.startswith("stream32_")


def requested_workers(plan):
    if not is_streaming(plan):
        return 1 if plan in S8_PLAN.values() else 6
    if plan.endswith("hvx2"):
        return 2
    if plan.endswith("hvx3"):
        return 3
    if plan.endswith("hvx4"):
        return 4
    return 6


def assert_zero(record, fields):
    for field in fields:
        if record[field] != 0:
            raise SystemExit(f"{field} failure for {key(record)}: {record[field]}")


def assert_runtime_valid(record, repeat):
    shape, plan, _ = key(record)
    if shape not in SHAPES or plan not in PLANS[shape]:
        raise SystemExit(f"unknown shape/plan for {key(record)}")
    pair = shape == "gate_up_pair"
    s8 = plan == S8_PLAN[shape]
    streaming = is_streaming(plan)
    w4 = not s8
    n_tiles = 384 if pair else 64
    k_tiles = 64 if pair else 192
    hmx_pairs = 24576 if pair else 12288
    hmx_streams = 768 if pair else 384
    parent_chunks = 1 if pair else 2
    expected_expands = (
        0 if s8 else
        hmx_streams * repeat if streaming else
        n_tiles * parent_chunks * repeat
    )

    if record["experiment"] != "EXP-0019" or record["repeat_count"] != repeat:
        raise SystemExit(f"wrong experiment/repeat for {key(record)}")
    expected_storage = "expanded_s8_control" if s8 else "packed_w4_hmx_postscale"
    if record["weight_storage"] != expected_storage:
        raise SystemExit(f"wrong storage for {key(record)}")
    if record["requested_hvx_workers"] != requested_workers(plan):
        raise SystemExit(f"wrong worker request for {key(record)}")
    if (record["host_invocation_mode"] != "single_invocation" or
            record["measured_rpc_calls"] != 1 or
            record["resource_lifetime_mode"] != "prepared_session" or
            record["output_assembly_mode"] != "linked_2d_dma"):
        raise SystemExit(f"wrong execution boundary for {key(record)}")
    if (record["rpc_result"] != 0 or record["warmup_rpc_result"] != 0 or
            record["dsp_status"] != 0 or record["mismatches"] != 0):
        raise SystemExit(f"runtime/correctness failure for {key(record)}")
    if record["warmup_output_checksum"] != record["measured_output_checksum"]:
        raise SystemExit(f"warm-up output changed for {key(record)}")
    assert_zero(record, (
        "prepare_result", "release_result", "session_close_result",
        "output_dma_status", "dcvs_power_setup_status",
        "dcvs_power_reset_status", "hmx_resource_status", "hmx_lock_status",
        "hmx_unlock_status", "hmx_release_status", "hmx_thread_create_status",
        "hmx_thread_join_status", "hvx_lock_status", "hvx_unlock_status",
        "dma_status", "sync_status", "dma_descriptor_timeout_count",
        "output_dma_descriptor_timeout_count", "streaming_ready_timeout_count",
    ))

    expected = {
        "projection_m": 64,
        "projection_k": 2048 if pair else 6144,
        "projection_n": 12288 if pair else 2048,
        "k_tile_count": k_tiles,
        "n_tile_count": n_tiles,
        "hmx_execution_count": hmx_pairs * repeat,
        "hmx_stream_count": hmx_streams * repeat,
        "weight_bundle_stage_count": n_tiles * repeat,
        "output_tile_count": n_tiles * repeat,
        "activation_stage_count": k_tiles,
        "weight_expand_count": expected_expands,
        "chunk_expand_count": expected_expands,
        "output_dma_submit_count": 1,
        "output_dma_wait_count": 2,
        "output_dma_descriptor_count": n_tiles,
        "output_dma_chain_count": 1,
        "output_dma_descriptor_completion_count": n_tiles,
        "resource_setup_in_run": 0,
        "resource_release_in_run": 0,
        "warmup_prepared_session_run_index": 1,
        "prepared_session_run_index": 2,
        "streaming_region_publish_count": hmx_streams * repeat if streaming else 0,
    }
    for field, value in expected.items():
        if record[field] != value:
            raise SystemExit(
                f"wrong {field} for {key(record)}: {record[field]} != {value}")

    if (record["vtcm_requested_bytes"] != 2097152 or
            record["vtcm_acquired_bytes"] != 2097152 or
            record["vtcm_plan_bytes"] > 2097152):
        raise SystemExit(f"VTCM contract failure for {key(record)}")
    if (record["resource_vtcm_address"] == 0 or
            record["resource_hmx_context_id"] == 0 or
            record["warmup_resource_vtcm_address"] != record["resource_vtcm_address"] or
            record["warmup_resource_hmx_context_id"] != record["resource_hmx_context_id"]):
        raise SystemExit(f"resource identity failure for {key(record)}")
    if (record["prepare_wall_ns"] <= 0 or record["release_wall_ns"] <= 0 or
            record["first_call_host_wall_ns"] <= 0 or
            record["second_call_host_wall_ns"] != 0 or
            record["pipeline_ticks"] <= 0 or record["hmx_compute_ticks"] <= 0):
        raise SystemExit(f"missing lifecycle/hot timing for {key(record)}")

    if w4:
        workers = requested_workers(plan)
        if (record["hvx_workers_created"] != workers or
                record["hvx_workers_locked"] != workers or
                record["hvx_hmx_overlap_observed"] != 1 or
                record["hvx_parallel_overlap_observed"] != 1):
            raise SystemExit(f"worker/overlap failure for {key(record)}")
        if not 1 <= record["hvx_max_active_workers"] <= workers:
            raise SystemExit(f"invalid active-worker count for {key(record)}")
        if record["hmx_carrier_checksum"] != record["packed_w4_checksum"]:
            raise SystemExit(f"W4 carrier failure for {key(record)}")
    elif record["hmx_carrier_checksum"] != record["expanded_carrier_checksum"]:
        raise SystemExit(f"S8 carrier failure for {key(record)}")


def group_records(records):
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def assert_matrix(groups, records_per_key, repeat):
    expected = expected_keys()
    if set(groups) != expected:
        raise SystemExit(
            f"matrix keys mismatch missing={expected - set(groups)} "
            f"extra={set(groups) - expected}")
    if any(len(group) != records_per_key for group in groups.values()):
        raise SystemExit("matrix has wrong records per key")
    for group in groups.values():
        for record in group:
            assert_runtime_valid(record, repeat)


def assert_equivalent(groups):
    for shape in SHAPES:
        for pattern in PATTERNS:
            checksums = {
                (record["reference_checksum"], record["measured_output_checksum"])
                for plan in PLANS[shape]
                for record in groups[(shape, plan, pattern)]
            }
            if len(checksums) != 1:
                raise SystemExit(f"variant output mismatch for {shape}/{pattern}")


def median_metrics(records):
    fields = (
        "host_wall_ns", "dsp_total_ticks", "pipeline_ticks",
        "activation_stage_ticks", "weight_stage_ticks", "weight_expand_ticks",
        "hmx_compute_ticks", "hmx_ready_wait_ticks",
        "producer_slot_wait_ticks", "expanded_slot_wait_ticks",
        "output_assembly_ticks", "input_cache_ticks", "output_cache_ticks",
        "vtcm_plan_bytes", "hvx_max_active_workers",
    )
    return {
        f"{field}_median": statistics.median(record[field] for record in records)
        for field in fields
    }


def summarize(groups):
    metrics = {}
    comparisons = {}
    for shape in SHAPES:
        for plan in PLANS[shape]:
            records = [
                record
                for (group_shape, group_plan, _), group in groups.items()
                if group_shape == shape and group_plan == plan
                for record in group
            ]
            metrics[f"{shape}:{plan}"] = median_metrics(records)

        parent_plan = PARENT_PLAN[shape]
        parent = metrics[f"{shape}:{parent_plan}"]["host_wall_ns_median"]
        s8_plan = S8_PLAN[shape]
        s8 = metrics[f"{shape}:{s8_plan}"]["host_wall_ns_median"]
        candidates = {
            plan: metrics[f"{shape}:{plan}"]["host_wall_ns_median"]
            for plan in STREAMING_PLANS[shape]
        }
        selected_plan = min(candidates, key=candidates.get)
        selected = candidates[selected_plan]
        comparisons[shape] = {
            "parent_plan": parent_plan,
            "parent_host_wall_ns_median": parent,
            "expanded_s8_plan": s8_plan,
            "expanded_s8_host_wall_ns_median": s8,
            "streaming_candidates_host_wall_ns_median": candidates,
            "fastest_streaming_plan": selected_plan,
            "fastest_streaming_host_wall_ns_median": selected,
            "streaming_vs_parent_improvement_percent":
                (parent - selected) * 100.0 / parent,
            "fastest_streaming_beats_parent": selected < parent,
            "fastest_streaming_vs_s8_improvement_percent":
                (s8 - selected) * 100.0 / s8,
        }
    return metrics, comparisons


def joint_selection(repeat1_metrics, repeat10_metrics):
    selections = {}
    for shape in SHAPES:
        parent_plan = PARENT_PLAN[shape]
        parent1 = repeat1_metrics[f"{shape}:{parent_plan}"]["host_wall_ns_median"]
        parent10 = repeat10_metrics[f"{shape}:{parent_plan}"]["host_wall_ns_median"]
        candidates = []
        for plan in STREAMING_PLANS[shape]:
            value1 = repeat1_metrics[f"{shape}:{plan}"]["host_wall_ns_median"]
            value10 = repeat10_metrics[f"{shape}:{plan}"]["host_wall_ns_median"]
            ratio1 = value1 / parent1
            ratio10 = value10 / parent10
            if ratio1 < 1.0 and ratio10 < 1.0:
                candidates.append((max(ratio1, ratio10), plan, ratio1, ratio10))
        candidates.sort()
        if candidates:
            _, plan, ratio1, ratio10 = candidates[0]
            selections[shape] = {
                "joint_gate": True,
                "selected_plan": plan,
                "repeat1_improvement_percent": (1.0 - ratio1) * 100.0,
                "repeat10_improvement_percent": (1.0 - ratio10) * 100.0,
            }
        else:
            selections[shape] = {"joint_gate": False, "selected_plan": None}
    return selections


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    result_dir = pathlib.Path(sys.argv[1])
    correctness = load_jsonl(result_dir / "correctness_repeat1.jsonl")
    timing1 = load_jsonl(result_dir / "timing_repeat1.jsonl")
    timing10 = load_jsonl(result_dir / "timing_repeat10.jsonl")
    correctness_groups = group_records(correctness)
    timing1_groups = group_records(timing1)
    timing10_groups = group_records(timing10)
    assert_matrix(correctness_groups, 1, 1)
    assert_matrix(timing1_groups, 5, 1)
    assert_matrix(timing10_groups, 5, 10)
    for groups in (correctness_groups, timing1_groups, timing10_groups):
        assert_equivalent(groups)

    repeat1_metrics, repeat1_comparisons = summarize(timing1_groups)
    repeat10_metrics, repeat10_comparisons = summarize(timing10_groups)
    selections = joint_selection(repeat1_metrics, repeat10_metrics)
    local_gate = any(value["joint_gate"] for value in selections.values())
    print(json.dumps({
        "experiment": "EXP-0019",
        "records": len(correctness) + len(timing1) + len(timing10),
        "correctness_records": len(correctness),
        "timing_repeat1_records": len(timing1),
        "timing_repeat10_records": len(timing10),
        "exact_reference": True,
        "all_physical_plans_equivalent": True,
        "persistent_resource_identity_stable": True,
        "worker_fallback": False,
        "readiness_timeouts": 0,
        "device_reboot": False,
        "vtcm_request_bytes": 2097152,
        "local_performance_gate": local_gate,
        "joint_selection": selections,
        "repeat1_comparisons": repeat1_comparisons,
        "repeat10_comparisons": repeat10_comparisons,
        "repeat1_metrics": repeat1_metrics,
        "repeat10_metrics": repeat10_metrics,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
