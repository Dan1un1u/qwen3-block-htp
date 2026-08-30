#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics


EXPECTED_HASH = {
    "F16F16": "704252c89780e695",
    "W4F16": "f18b9abbe1487231",
    "W4U8": "69f22eeb035e5ec5",
}
SCHEDULES = (
    "control",
    "group_major",
    "group_major2",
    "group_major4",
    "q_prefix4_k_all",
    "q_prefix6_k_all",
)
VARIANTS = ("F16F16", "W4F16", "W4U8")
PARITY_FIELDS = (
    "output_hash",
    "weight_ddr_read_bytes",
    "weight_dma_descriptor_count",
    "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes",
    "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes",
    "intermediate_dma_descriptor_count",
    "intermediate_spill_fill_count",
    "hmx_command_count",
    "hmx_fp16_tile_pair_count",
    "hmx_u8s8_tile_pair_count",
)


def load_single_json_line(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"expected one JSON line in {path}")
    return json.loads(lines[0])


def median(values):
    return float(statistics.median(values))


def validate_record(record, variant):
    if record["experiment"] != "EXP-0085" or record["variant"] != variant:
        raise SystemExit(f"wrong record identity for {variant}")
    if (record["output_hash"] != EXPECTED_HASH[variant] or
            record["mismatches"] != 0 or record["max_lsb"] != 0):
        raise SystemExit(f"correctness failure for {variant}")
    if (record["intermediate_ddr_read_bytes"] != 0 or
            record["intermediate_ddr_write_bytes"] != 0 or
            record["intermediate_spill_fill_count"] != 0 or
            record["vtcm_acquired_bytes"] != 8388608):
        raise SystemExit(f"physical gate failure for {variant}")


def validate_timeline(timeline, variant, schedule):
    if (timeline["experiment"] != "EXP-0085" or
            timeline["variant"] != variant or
            timeline["qkv_schedule"] != schedule):
        raise SystemExit(f"wrong timeline identity for {variant}/{schedule}")
    groups = timeline["groups"]
    if len(groups) != 8:
        raise SystemExit(f"expected eight groups for {variant}/{schedule}")
    required = (
        "q_projection_ready", "k_projection_ready", "v_projection_ready",
        "q_prep_ready", "k_prep_ready", "attention_consume")
    for group in groups:
        if any(group[key] <= 0 for key in required):
            raise SystemExit(
                f"incomplete group {group['group']} for {variant}/{schedule}")


def collect(root):
    round_dirs = sorted(path for path in root.glob("round_*") if path.is_dir())
    if len(round_dirs) < 3:
        raise SystemExit("Stage B requires at least three interleaved rounds")
    data = {variant: {schedule: [] for schedule in SCHEDULES}
            for variant in VARIANTS}
    for round_index, round_dir in enumerate(round_dirs):
        controls = {}
        for variant in VARIANTS:
            stem = f"control_{variant.lower()}"
            controls[variant] = load_single_json_line(
                round_dir / f"{stem}.jsonl")
        for variant in VARIANTS:
            for schedule in SCHEDULES:
                stem = f"{schedule}_{variant.lower()}"
                record = load_single_json_line(round_dir / f"{stem}.jsonl")
                timeline = json.loads(
                    (round_dir / f"{stem}_timeline.json").read_text())
                validate_record(record, variant)
                validate_timeline(timeline, variant, schedule)
                if any(record[field] != controls[variant][field]
                       for field in PARITY_FIELDS):
                    raise SystemExit(
                        f"physical parity failure for round {round_index} "
                        f"{variant}/{schedule}")
                groups = timeline["groups"]
                projection_tail = max(
                    max(group["q_projection_ready"],
                        group["k_projection_ready"],
                        group["v_projection_ready"])
                    for group in groups)
                prep_tail = max(
                    max(group["q_prep_ready"], group["k_prep_ready"])
                    for group in groups)
                data[variant][schedule].append({
                    "round": round_index,
                    "host_wall_ns": record["host_wall_ns_per_block"],
                    "qkv_ticks": record["qkv_projection_ticks"],
                    "timeline_qkv_ticks": timeline["qkv_end_ticks"],
                    "projection_tail_ticks": projection_tail,
                    "prep_tail_ticks": prep_tail,
                    "post_projection_drain_ticks":
                        timeline["qkv_end_ticks"] - projection_tail,
                    "early_publication_groups": sum(
                        min(group["q_projection_ready"],
                            group["k_projection_ready"],
                            group["v_projection_ready"])
                        < max(group["q_projection_ready"],
                              group["k_projection_ready"],
                              group["v_projection_ready"])
                        for group in groups),
                })
    return data


def summarize(data):
    cases = []
    for variant in VARIANTS:
        control = data[variant]["control"]
        control_qkv = median([item["qkv_ticks"] for item in control])
        control_host = median([item["host_wall_ns"] for item in control])
        for schedule in SCHEDULES:
            rows = data[variant][schedule]
            qkv = median([item["qkv_ticks"] for item in rows])
            host = median([item["host_wall_ns"] for item in rows])
            paired_qkv = median([
                control[index]["qkv_ticks"] - rows[index]["qkv_ticks"]
                for index in range(len(rows))])
            paired_host = median([
                control[index]["host_wall_ns"] - rows[index]["host_wall_ns"]
                for index in range(len(rows))])
            cases.append({
                "variant": variant,
                "schedule": schedule,
                "rounds": len(rows),
                "qkv_ticks_median": qkv,
                "qkv_vs_control_percent":
                    (control_qkv / qkv - 1.0) * 100.0,
                "qkv_paired_improvement_ticks_median": paired_qkv,
                "host_wall_ns_median": host,
                "host_vs_control_percent":
                    (control_host / host - 1.0) * 100.0,
                "host_paired_improvement_ns_median": paired_host,
                "projection_tail_ticks_median": median([
                    item["projection_tail_ticks"] for item in rows]),
                "prep_tail_ticks_median": median([
                    item["prep_tail_ticks"] for item in rows]),
                "post_projection_drain_ticks_median": median([
                    item["post_projection_drain_ticks"] for item in rows]),
                "early_publication_groups_min": min(
                    item["early_publication_groups"] for item in rows),
            })

    by_case = {(item["variant"], item["schedule"]): item for item in cases}
    selectable = []
    for schedule in SCHEDULES[1:]:
        f16 = by_case[("F16F16", schedule)]
        w4f16 = by_case[("W4F16", schedule)]
        w4u8 = by_case[("W4U8", schedule)]
        if (w4f16["qkv_vs_control_percent"] > 0 and
                w4f16["host_vs_control_percent"] > 0 and
                w4f16["qkv_paired_improvement_ticks_median"] > 0 and
                w4f16["host_paired_improvement_ns_median"] > 0 and
                w4u8["qkv_vs_control_percent"] > 0 and
                w4u8["host_vs_control_percent"] > 0 and
                w4u8["qkv_paired_improvement_ticks_median"] > 0 and
                w4u8["host_paired_improvement_ns_median"] > 0 and
                f16["qkv_vs_control_percent"] >= 0 and
                f16["host_vs_control_percent"] >= 0):
            selectable.append(schedule)
    return {
        "experiment": "EXP-0085",
        "stage": "B",
        "candidate_selection_gate": "pass" if selectable else "fail",
        "selectable_schedules": selectable,
        "cases": cases,
    }


def render_markdown(summary):
    print("# EXP-0085 Stage B bounded schedule search")
    print()
    print(f"Candidate selection gate: **{summary['candidate_selection_gate']}**")
    print()
    print("| Variant | Schedule | QKV ticks | QKV vs control | "
          "Host wall | Host vs control | Projection tail | Prep tail |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in summary["cases"]:
        print(
            f"| {item['variant']} | {item['schedule']} | "
            f"{item['qkv_ticks_median']:.0f} | "
            f"{item['qkv_vs_control_percent']:+.2f}% | "
            f"{item['host_wall_ns_median'] / 1000.0:.1f} us | "
            f"{item['host_vs_control_percent']:+.2f}% | "
            f"{item['projection_tail_ticks_median']:.0f} | "
            f"{item['prep_tail_ticks_median']:.0f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("root", type=pathlib.Path)
    args = parser.parse_args()
    summary = summarize(collect(args.root))
    if args.markdown:
        render_markdown(summary)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
