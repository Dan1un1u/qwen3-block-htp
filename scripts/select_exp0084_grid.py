#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


VARIANTS = ("f16f16", "w4f16")
REPEATS = (1, 10)
ROWS = (2, 4, 8)
CONTEXTS = (2, 3, 4)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


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
                    if not records:
                        raise SystemExit(f"empty grid cell: {path}")
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
    result = {
        "selection_rule": (
            "minimize maximum normalized host-wall regret across "
            "F16F16/W4F16 and repeat1/repeat10; break ties by mean regret"
        ),
        "selected_rows_per_task": selected["rows_per_task"],
        "selected_contexts": selected["contexts"],
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
