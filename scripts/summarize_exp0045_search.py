#!/usr/bin/env python3
"""Summarize an interleaved EXP-0045 batch-size search."""

from __future__ import annotations

import json
import pathlib
import statistics
import sys


FIELDS = (
    "host_wall_ns_per_block",
    "total_ticks",
    "qkv_projection_ticks",
    "attention_ticks",
    "attention_qk_norm_pool_wait_ticks",
    "hmx_command_count",
    "weight_dma_descriptor_count",
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    root = pathlib.Path(sys.argv[1])
    summary: dict[str, object] = {"result_dir": str(root), "modes": {}}
    for repeat in (1, 10):
        repeat_summary: dict[str, object] = {}
        for mode in ("serial", "qkv_batch2", "qkv_batch4"):
            path = root / f"{mode}_r{repeat}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if not records:
                raise SystemExit(f"no records: {path}")
            for record in records:
                if record["w4u8_qkvo_pipeline_mode"] != mode:
                    raise SystemExit(f"wrong mode in {path}")
                if record["mismatches"] != 0 or record["max_lsb"] != 0:
                    raise SystemExit(f"numerical mismatch in {path}")
            mode_summary: dict[str, object] = {"samples": len(records)}
            for field in FIELDS:
                values = []
                for record in records:
                    value = float(record[field])
                    if field != "host_wall_ns_per_block":
                        value /= int(record["repeat_count"])
                    values.append(value)
                mode_summary[f"{field}_median"] = statistics.median(values)
            repeat_summary[mode] = mode_summary
        summary["modes"][f"repeat{repeat}"] = repeat_summary
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
