#!/usr/bin/env python3
"""Verify repeated EXP-0164 device generation logs and decode the tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--semantic-reference",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0164/"
            "semantic_gate/teacher_w4f16_greedy16.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    semantic = json.loads(
        args.semantic_reference.read_text(encoding="utf-8")
    )
    expected = semantic["w4f16"]["token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    runs: list[dict[str, object]] = []

    for path in args.logs:
        records = read_records(path)
        steps = [r for r in records if "generation_step" in r]
        finals = [
            r for r in records if r.get("generation_sequence_complete")
        ]
        if len(steps) != 16 or len(finals) != 1:
            raise ValueError(f"{path}: expected 16 steps and one final record")
        actual = [int(step["selected_token_id"]) for step in steps]
        for index, step in enumerate(steps):
            required = (
                int(step["generation_step"]) == index
                and bool(step["token_match"])
                and bool(step["pass"])
                and int(step["boundary_ddr_write_bytes"]) == 0
                and int(step["intermediate_ddr_read_bytes"]) == 0
                and int(step["intermediate_ddr_write_bytes"]) == 0
                and int(step["intermediate_spill_fill_count"]) == 0
                and int(step["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                and int(step["rpc_result"]) == 0
            )
            if not required:
                raise ValueError(f"{path}: device gate failed at step {index}")
        if actual != expected or finals[0]["token_ids"] != expected:
            raise ValueError(f"{path}: token sequence mismatch")
        if not finals[0]["all_steps_pass"]:
            raise ValueError(f"{path}: final gate is false")
        prefill_ns = int(steps[0]["host_wall_ns"])
        decode_values = [int(step["host_wall_ns"]) for step in steps[1:]]
        runs.append(
            {
                "log": str(path.resolve()),
                "sha256": sha256_file(path),
                "token_ids": actual,
                "decoded_text": tokenizer.decode(
                    actual, skip_special_tokens=True
                ),
                "prefill_host_wall_us": prefill_ns / 1000.0,
                "decode_mean_host_wall_us": (
                    sum(decode_values) / len(decode_values) / 1000.0
                ),
                "total_host_wall_us": int(
                    finals[0]["total_host_wall_ns"]
                )
                / 1000.0,
            }
        )

    result = {
        "experiment": "EXP-0164",
        "variant": "W4F16",
        "run_count": len(runs),
        "all_runs_exact": True,
        "independent_expected_token_ids": expected,
        "independent_expected_text": semantic["w4f16"]["text"],
        "device_decoded_text": runs[0]["decoded_text"],
        "stable_across_runs": all(
            run["token_ids"] == runs[0]["token_ids"] for run in runs
        ),
        "physical_gate": {
            "vtcm_bytes": 8 * 1024 * 1024,
            "intermediate_ddr_read_bytes": 0,
            "intermediate_ddr_write_bytes": 0,
            "timed_full_logits_ddr_write_bytes": 0,
            "spill_fill_count": 0,
            "fastrpc_calls_per_pass": 1,
        },
        "runs": runs,
    }
    payload = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
