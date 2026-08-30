#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


VARIANTS = ("f16f16", "w4f16")
REPEATS = (1, 10)
ROWS = (2, 4, 8)
CONTEXTS = (2, 3, 4)
EXPECTED_SAMPLES = 2
EXPECTED_HASHES = {
    "f16f16": "704252c89780e695",
    "w4f16": "f18b9abbe1487231",
}


def load_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if len(records) != EXPECTED_SAMPLES:
        raise SystemExit(
            f"{path}: expected {EXPECTED_SAMPLES} records, got {len(records)}")
    return records


def validate_record(record: dict, variant: str, repeat: int) -> None:
    fixed = {
        "experiment": "EXP-0084",
        "variant": variant.upper(),
        "repeat_count": repeat,
        "rpc_result": 0,
        "dsp_status": 3,
        "numerical_status": 1,
        "intermediate_residency": "VTCM",
        "vtcm_requested_bytes": 8_388_608,
        "vtcm_acquired_bytes": 8_388_608,
        "intermediate_ddr_read_bytes": 0,
        "intermediate_ddr_write_bytes": 0,
        "intermediate_spill_fill_count": 0,
        "output_hash": EXPECTED_HASHES[variant],
    }
    for field, expected in fixed.items():
        if record.get(field) != expected:
            raise SystemExit(
                f"{variant} repeat{repeat} {field}: "
                f"{record.get(field)!r} != {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()

    values: dict[tuple[int, int], list[float]] = {}
    for rows in ROWS:
        for contexts in CONTEXTS:
            cells = []
            for variant in VARIANTS:
                for repeat in REPEATS:
                    path = args.grid_dir / (
                        f"{variant}_r{rows}_c{contexts}_repeat{repeat}.jsonl")
                    records = load_jsonl(path)
                    for record in records:
                        validate_record(record, variant, repeat)
                    cells.append(statistics.median(
                        record["host_wall_ns_per_block"]
                        for record in records))
            values[(rows, contexts)] = cells

    cell_minima = [
        min(candidate[index] for candidate in values.values())
        for index in range(4)
    ]
    scored = []
    for (rows, contexts), cells in values.items():
        relative = [cells[index] / cell_minima[index] - 1.0
                    for index in range(4)]
        scored.append({
            "rows_per_task": rows,
            "contexts": contexts,
            "host_wall_ns_per_block": {
                "f16f16_repeat1": cells[0],
                "f16f16_repeat10": cells[1],
                "w4f16_repeat1": cells[2],
                "w4f16_repeat10": cells[3],
            },
            "maximum_relative_to_cell_best": max(relative),
            "mean_relative_to_cell_best": statistics.mean(relative),
        })
    scored.sort(key=lambda item: (
        item["maximum_relative_to_cell_best"],
        item["mean_relative_to_cell_best"],
        item["rows_per_task"], item["contexts"]))
    selected = scored[0]
    recipe_best = {}
    for variant, indices in (("f16f16", (0, 1)), ("w4f16", (2, 3))):
        recipe_scored = []
        for candidate in scored:
            cells = (
                candidate["host_wall_ns_per_block"][
                    f"{variant}_repeat1"],
                candidate["host_wall_ns_per_block"][
                    f"{variant}_repeat10"],
            )
            relative = [
                cells[index] / cell_minima[indices[index]] - 1.0
                for index in range(2)
            ]
            recipe_scored.append({
                "rows_per_task": candidate["rows_per_task"],
                "contexts": candidate["contexts"],
                "host_wall_ns_per_block": {
                    "repeat1": cells[0],
                    "repeat10": cells[1],
                },
                "maximum_relative_to_recipe_cell_best": max(relative),
                "mean_relative_to_recipe_cell_best": statistics.mean(relative),
            })
        recipe_scored.sort(key=lambda item: (
            item["maximum_relative_to_recipe_cell_best"],
            item["mean_relative_to_recipe_cell_best"],
            item["rows_per_task"], item["contexts"]))
        recipe_best[variant] = recipe_scored[0]
    result = {
        "selection_rule": (
            "minimize maximum normalized host-wall regret across "
            "F16F16/W4F16 and repeat1/repeat10; break ties by mean regret"
        ),
        "selected_rows_per_task": selected["rows_per_task"],
        "selected_contexts": selected["contexts"],
        "recipe_specific_best": recipe_best,
        "candidates": scored,
    }
    if args.plain:
        print(result["selected_rows_per_task"],
              result["selected_contexts"])
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
