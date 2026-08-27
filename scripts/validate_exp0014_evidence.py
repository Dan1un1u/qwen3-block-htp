#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


PATTERNS = ("identity", "signed", "structured", "boundary")
PROJECTION_CONFIG = {
    "gate_up": {"k_tiles": 64, "n_tiles": 192},
    "down": {"k_tiles": 192, "n_tiles": 64},
}


def variants(projection):
    if projection == "gate_up":
        plans = (
            ("expanded_s8_control", "expanded_s8_dma_chain2",
             1, 2, 8, 32, 2, 2, True),
            ("packed_w4_hmx_postscale",
             "slots8e7_chunk64_dma_chain4",
             6, 8, 7, 64, 4, 1, True),
        )
    else:
        plans = (
            ("expanded_s8_control", "exp0005_full_bundle_control",
             1, 2, 8, 32, 1, 6, False),
            ("packed_w4_hmx_postscale",
             "slots4_chunk96_dma_batch2",
             6, 4, 8, 96, 2, 2, False),
        )
    return tuple(
        (*plan, output_mode)
        for plan in plans
        for output_mode in ("scalar_memcpy", "linked_2d_dma")
    )


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
    return (record["weight_storage"], record["physical_plan"],
            record["output_assembly_mode"], record["projection"],
            record["pattern"])


def plan_config(record):
    for variant in variants(record["projection"]):
        if (record["weight_storage"], record["physical_plan"],
                record["output_assembly_mode"]) == (
                    variant[0], variant[1], variant[9]):
            return variant
    raise SystemExit(f"unknown variant: {key(record)}")


def assert_runtime_valid(record, repeat):
    record_key = key(record)
    if record["experiment"] != "EXP-0014":
        raise SystemExit(f"wrong experiment for {record_key}")
    if record["repeat_count"] != repeat:
        raise SystemExit(f"wrong repeat for {record_key}")
    if record["rpc_result"] != 0 or record["dsp_status"] != 0:
        raise SystemExit(f"runtime failure for {record_key}")
    if record["mismatches"] != 0:
        raise SystemExit(f"output mismatch for {record_key}")
    if record["dma_status"] != 0 or record["sync_status"] != 0:
        raise SystemExit(f"DMA/sync failure for {record_key}")
    for field in (
            "dcvs_power_setup_status", "dcvs_power_reset_status",
            "hmx_resource_status", "hmx_lock_status", "hmx_unlock_status",
            "hmx_release_status", "hmx_thread_create_status",
            "hmx_thread_join_status", "hvx_lock_status",
            "hvx_unlock_status", "output_dma_status"):
        if record[field] != 0:
            raise SystemExit(f"{field} failure for {record_key}")

    (storage, _, workers, compressed, expanded, chunk, batch, chunks,
     linked, output_mode) = plan_config(record)
    config = PROJECTION_CONFIG[record["projection"]]
    logical_bundles = config["n_tiles"] * repeat
    weight_groups = logical_bundles // batch
    activation_descriptors = config["k_tiles"]
    expected_submits = activation_descriptors + weight_groups
    expected_descriptors = activation_descriptors + (
        logical_bundles if linked else weight_groups)
    expected_chains = weight_groups if linked else 0
    expected_waits = (2 * activation_descriptors + logical_bundles +
                      2 * weight_groups if linked else 2 * expected_submits)
    expected_expands = logical_bundles * chunks if storage == \
        "packed_w4_hmx_postscale" else 0
    output_dma = output_mode == "linked_2d_dma"

    expected_fields = {
        "requested_hvx_workers": workers,
        "compressed_slot_count": compressed,
        "expanded_chunk_slot_count": expanded,
        "chunk_tiles": chunk,
        "dma_bundle_batch": batch,
        "chunks_per_output": chunks,
        "hmx_execution_count": 12288 * repeat,
        "hmx_stream_count": 384 * repeat,
        "weight_bundle_stage_count": logical_bundles,
        "output_tile_count": logical_bundles,
        "weight_expand_count": expected_expands,
        "chunk_expand_count": expected_expands,
        "dma_submit_count": expected_submits,
        "dma_wait_count": expected_waits,
        "dma_descriptor_count": expected_descriptors,
        "dma_chain_count": expected_chains,
        "dma_descriptor_completion_count": expected_descriptors,
        "dma_descriptor_timeout_count": 0,
        "output_dma_submit_count": 1 if output_dma else 0,
        "output_dma_wait_count": 2 if output_dma else 0,
        "output_dma_descriptor_count": (
            config["n_tiles"] if output_dma else 0),
        "output_dma_chain_count": 1 if output_dma else 0,
        "output_dma_descriptor_completion_count": (
            config["n_tiles"] if output_dma else 0),
        "output_dma_descriptor_timeout_count": 0,
        "weight_slot_reuse_count": max(0, logical_bundles - compressed),
        "expanded_chunk_slot_reuse_count": (
            max(0, expected_expands - expanded)
            if storage == "packed_w4_hmx_postscale" else 0),
    }
    for field, expected in expected_fields.items():
        if record[field] != expected:
            raise SystemExit(
                f"wrong {field} for {record_key}: "
                f"{record[field]} != {expected}")
    if record["vtcm_requested_bytes"] != 2097152 or \
            record["vtcm_acquired_bytes"] != 2097152 or \
            record["vtcm_plan_bytes"] > 2097152:
        raise SystemExit(f"VTCM contract failure for {record_key}")

    if storage == "packed_w4_hmx_postscale":
        if record["hvx_workers_created"] != workers or \
                record["hvx_workers_locked"] != workers:
            raise SystemExit(f"worker fallback for {record_key}")
        if record["hvx_hmx_overlap_observed"] != 1 or \
                record["hvx_parallel_overlap_observed"] != 1:
            raise SystemExit(f"missing HVX/HMX overlap for {record_key}")
        if record["hmx_carrier_checksum"] != record["packed_w4_checksum"]:
            raise SystemExit(f"HMX carrier is not raw q4 for {record_key}")
    elif record["hmx_carrier_checksum"] != record[
            "expanded_carrier_checksum"]:
        raise SystemExit(f"S8 carrier mismatch for {record_key}")


def group_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record)
    return grouped


def median_metrics(records):
    fields = (
        "host_wall_ns", "dsp_total_ticks", "pipeline_ticks",
        "output_assembly_ticks", "weight_stage_ticks",
        "weight_expand_ticks", "hmx_compute_ticks",
        "hmx_ready_wait_ticks", "producer_slot_wait_ticks",
        "expanded_slot_wait_ticks", "dma_submit_count", "dma_wait_count",
        "dma_descriptor_count", "dma_chain_count",
        "output_dma_descriptor_count", "hvx_max_active_workers",
        "vtcm_plan_bytes",
    )
    return {
        f"{field}_median": statistics.median(
            record[field] for record in records)
        for field in fields
    }


def expected_keys():
    return {
        (variant[0], variant[1], variant[9], projection, pattern)
        for projection in PROJECTION_CONFIG
        for variant in variants(projection)
        for pattern in PATTERNS
    }


def assert_matrix(groups, records_per_key, repeat):
    if set(groups) != expected_keys():
        raise SystemExit("matrix has missing/unexpected keys")
    if any(len(group) != records_per_key for group in groups.values()):
        raise SystemExit("matrix has wrong records per key")
    for group in groups.values():
        for record in group:
            assert_runtime_valid(record, repeat)


def assert_equivalent(groups):
    comparable = (
        "reference_checksum", "reference_min", "reference_max",
        "expanded_carrier_checksum", "packed_w4_checksum", "projection_m",
        "projection_k", "projection_n", "hmx_execution_count",
        "hmx_stream_count", "output_tile_count",
        "weight_bundle_stage_count", "activation_stage_count",
        "weight_expand_count", "dma_submit_count", "dma_wait_count",
        "dma_descriptor_count", "dma_chain_count",
    )
    for projection in PROJECTION_CONFIG:
        for pattern in PATTERNS:
            for storage in ("expanded_s8_control",
                            "packed_w4_hmx_postscale"):
                peers = [record for group_key, group in groups.items()
                         if group_key[0] == storage and
                         group_key[3:] == (projection, pattern)
                         for record in group]
                reference = peers[0]
                for record in peers[1:]:
                    for field in comparable:
                        if record[field] != reference[field]:
                            raise SystemExit(
                                f"{projection}/{pattern}/{storage}: "
                                f"output modes differ at {field}")


def summarize(groups):
    metrics = {}
    for projection in PROJECTION_CONFIG:
        for variant in variants(projection):
            storage, plan, *_, output_mode = variant
            records = [record for group_key, group in groups.items()
                       if group_key[:4] == (
                           storage, plan, output_mode, projection)
                       for record in group]
            metrics[f"{projection}:{storage}:{plan}:{output_mode}"] = \
                median_metrics(records)
    return metrics


def compare_modes(metrics, repeat_name):
    comparisons = {}
    w4_gate = True
    for projection in PROJECTION_CONFIG:
        for storage in ("expanded_s8_control",
                        "packed_w4_hmx_postscale"):
            plan = next(variant[1] for variant in variants(projection)
                        if variant[0] == storage)
            prefix = f"{projection}:{storage}:{plan}:"
            scalar = metrics[prefix + "scalar_memcpy"]
            dma = metrics[prefix + "linked_2d_dma"]
            scalar_wall = scalar["host_wall_ns_median"]
            dma_wall = dma["host_wall_ns_median"]
            comparison = {
                "scalar_host_wall_ns_median": scalar_wall,
                "dma_host_wall_ns_median": dma_wall,
                "host_wall_improvement_percent":
                    (scalar_wall - dma_wall) * 100.0 / scalar_wall,
                "scalar_output_assembly_ticks_median":
                    scalar["output_assembly_ticks_median"],
                "dma_output_assembly_ticks_median":
                    dma["output_assembly_ticks_median"],
                "output_stage_improvement_percent":
                    (scalar["output_assembly_ticks_median"] -
                     dma["output_assembly_ticks_median"]) * 100.0 /
                    scalar["output_assembly_ticks_median"],
            }
            comparisons[f"{projection}:{storage}"] = comparison
            if storage == "packed_w4_hmx_postscale":
                if repeat_name == "repeat1":
                    w4_gate &= dma_wall < scalar_wall
                else:
                    w4_gate &= dma_wall <= scalar_wall

        s8_plan = next(variant[1] for variant in variants(projection)
                       if variant[0] == "expanded_s8_control")
        w4_plan = next(variant[1] for variant in variants(projection)
                       if variant[0] == "packed_w4_hmx_postscale")
        s8 = metrics[
            f"{projection}:expanded_s8_control:{s8_plan}:linked_2d_dma"]
        w4 = metrics[
            f"{projection}:packed_w4_hmx_postscale:{w4_plan}:linked_2d_dma"]
        s8_wall = s8["host_wall_ns_median"]
        w4_wall = w4["host_wall_ns_median"]
        comparisons[f"{projection}:fair_storage_comparison"] = {
            "w4_host_wall_ns_median": w4_wall,
            "s8_host_wall_ns_median": s8_wall,
            "w4_vs_s8_improvement_percent":
                (s8_wall - w4_wall) * 100.0 / s8_wall,
            "w4_beats_s8": w4_wall < s8_wall,
        }
    return comparisons, w4_gate


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
    assert_matrix(timing1_groups, 3, 1)
    assert_matrix(timing10_groups, 3, 10)
    for groups in (correctness_groups, timing1_groups, timing10_groups):
        assert_equivalent(groups)

    repeat1_metrics = summarize(timing1_groups)
    repeat10_metrics = summarize(timing10_groups)
    repeat1_comparisons, repeat1_gate = compare_modes(
        repeat1_metrics, "repeat1")
    repeat10_comparisons, repeat10_gate = compare_modes(
        repeat10_metrics, "repeat10")
    fair_storage_gate = all(
        comparison["w4_beats_s8"]
        for label, comparison in repeat1_comparisons.items()
        if label.endswith("fair_storage_comparison"))

    print(json.dumps({
        "experiment": "EXP-0014",
        "records": len(correctness) + len(timing1) + len(timing10),
        "correctness_records": len(correctness),
        "timing_repeat1_records": len(timing1),
        "timing_repeat10_records": len(timing10),
        "exact_reference": True,
        "output_modes_pre_output_equivalent": True,
        "worker_fallback": False,
        "descriptor_timeouts": 0,
        "device_reboot": False,
        "vtcm_request_bytes": 2097152,
        "local_performance_gate": repeat1_gate and repeat10_gate,
        "repeat1_w4_gate": repeat1_gate,
        "repeat10_w4_nonregression_gate": repeat10_gate,
        "beats_fair_expanded_s8_repeat1": fair_storage_gate,
        "repeat1_comparisons": repeat1_comparisons,
        "repeat10_comparisons": repeat10_comparisons,
        "repeat1_metrics": repeat1_metrics,
        "repeat10_metrics": repeat10_metrics,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
