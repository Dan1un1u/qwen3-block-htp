#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys


EXPECTED_HASH = {
    "F16F16": "704252c89780e695",
    "W4F16": "f18b9abbe1487231",
    "W4U8": "69f22eeb035e5ec5",
}


def load_single_json_line(path: pathlib.Path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"expected one JSON line in {path}")
    return json.loads(lines[0])


def summarize(variant, record, timeline):
    if record["experiment"] != "EXP-0085" or record["variant"] != variant:
        raise SystemExit(f"wrong record identity for {variant}")
    if record["output_hash"] != EXPECTED_HASH[variant] or \
            record["mismatches"] != 0 or record["max_lsb"] != 0:
        raise SystemExit(f"correctness failure for {variant}")
    if record["intermediate_ddr_read_bytes"] != 0 or \
            record["intermediate_ddr_write_bytes"] != 0 or \
            record["intermediate_spill_fill_count"] != 0 or \
            record["vtcm_acquired_bytes"] != 8388608:
        raise SystemExit(f"physical gate failure for {variant}")
    groups = timeline["groups"]
    if len(groups) != 8:
        raise SystemExit(f"expected eight groups for {variant}")
    required = (
        "q_projection_ready", "k_projection_ready", "v_projection_ready",
        "q_prep_ready", "k_prep_ready", "attention_consume")
    for group in groups:
        if any(group[key] <= 0 for key in required):
            raise SystemExit(
                f"incomplete group {group['group']} timeline for {variant}")
    projection_tail = max(
        max(group["q_projection_ready"], group["k_projection_ready"],
            group["v_projection_ready"]) for group in groups)
    q_to_k = [
        group["k_projection_ready"] - group["q_projection_ready"]
        for group in groups]
    prep_tail = max(
        max(group["q_prep_ready"], group["k_prep_ready"])
        for group in groups)
    return {
        "variant": variant,
        "output_hash": record["output_hash"],
        "host_wall_ns_per_block": record["host_wall_ns_per_block"],
        "qkv_projection_ticks": record["qkv_projection_ticks"],
        "timeline_qkv_end_ticks": timeline["qkv_end_ticks"],
        "projection_tail_ticks": projection_tail,
        "post_projection_drain_ticks":
            timeline["qkv_end_ticks"] - projection_tail,
        "prep_tail_ticks": prep_tail,
        "q_to_k_gap_ticks_median": statistics.median(q_to_k),
        "q_to_k_gap_ticks_min": min(q_to_k),
        "q_to_k_gap_ticks_max": max(q_to_k),
        "q_group7_before_k_group0":
            groups[7]["q_projection_ready"] <
            groups[0]["k_projection_ready"],
        "weight_dma_ticks": timeline["weight_dma_ticks"],
        "weight_expand_ticks": timeline["weight_expand_ticks"],
        "hmx_wait_ticks": timeline["hmx_wait_ticks"],
        "first_attention_consume_ticks": min(
            group["attention_consume"] for group in groups),
    }


def main():
    root = pathlib.Path(sys.argv[1])
    summaries = []
    for variant in ("F16F16", "W4F16", "W4U8"):
        stem = variant.lower()
        record = load_single_json_line(root / f"{stem}.jsonl")
        timeline = json.loads((root / f"{stem}_timeline.json").read_text())
        summaries.append(summarize(variant, record, timeline))
    addressable = [
        item for item in summaries
        if item["q_group7_before_k_group0"] and
        item["post_projection_drain_ticks"] > 0
    ]
    result = {
        "experiment": "EXP-0085",
        "stage": "A",
        "diagnostic_gate": "pass" if len(addressable) >= 2 else "fail",
        "addressable_variants": [item["variant"] for item in addressable],
        "variants": summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["diagnostic_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
