#!/usr/bin/env python3
"""Validate and summarize EXP-0169 long continuous-decode attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import summarize_exp0164 as base  # noqa: E402
import summarize_exp0167 as exp167  # noqa: E402


STEPS = 193
LAYERS = 28
EXPECTED_TOKENS = [
    124491, 51272, 51272, 9092, 51272, 128014, 23186, 85301,
    23186, 23186, 23186, 23186, 105260, 37440, 23186, 5205,
]
EXPECTED_CODES = [
    159, 162, 160, 161, 161, 160, 162, 164,
    163, 165, 165, 160, 156, 156, 165, 158,
]
BUCKETS: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    (
        f"L{first}-{first + 31}",
        tuple(range(1 + first - 64, 1 + first - 64 + 32)),
    )
    for first in range(64, 256, 32)
)
ROWS = (
    "Token embedding",
    *(name for name, _ in base.BASE_LEDGER[:-2]),
    "Final model RMSNorm",
    "Streaming W4 LM head + greedy argmax",
    base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
    "True Host-DSP boundary", "Complete Host wall",
)
ATTENTION_TICK_FIELDS = (
    "u8_attention_qk_norm_rope_ticks",
    "u8_attention_k_pack_ticks",
    "u8_attention_v_pack_ticks",
    "u8_attention_qk_hmx_ticks",
    "u8_attention_softmax_ticks",
    "u8_attention_av_hmx_ticks",
    "u8_attention_av_requant_ticks",
    "u8_attention_pipeline_wait_ticks",
    "attention_unattributed_ticks",
)
CACHE_COUNTER_FIELDS = (
    "scan_cache_ddr_read_bytes", "scan_cache_ddr_write_bytes",
    "scan_cache_dma_descriptor_count", "u8_cache_segment_seal_count",
    "u8_cache_segment_sealed_bytes", "u8_cache_segment_tail_append_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0169-w4u8-long-decode-attribution",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(result_dir: Path) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob("generation_??.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten generation logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        records = base.read_json_lines(path)
        steps = [
            item for item in records
            if "generation_step" in item and "record" not in item
        ]
        profiles = [
            item for item in records
            if item.get("record") == "generation_profile"
        ]
        finals = [
            item for item in records
            if item.get("generation_sequence_complete") is True
        ]
        if len(steps) != STEPS or len(profiles) != STEPS or len(finals) != 1:
            raise ValueError(f"incomplete generation run: {path}")
        if not (
            finals[0].get("all_steps_pass") is True
            and int(finals[0].get("requested_steps", -1)) == STEPS
            and int(finals[0].get("completed_steps", -1)) == STEPS
        ):
            raise ValueError(f"failed final record: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if not (
                int(step.get("experiment", -1)) == 169
                and int(profile.get("experiment", -1)) == 169
                and int(step.get("generation_step", -1)) == index
                and int(profile.get("generation_step", -1)) == index
                and int(step.get("generation_mode", -1)) == 9
            ):
                raise ValueError(f"identity mismatch: {path}:{index}")
            profile["_step_record"] = step
            profile["_source_log"] = str(path)
        runs.append(profiles)
    return runs


def run_mean(
    run: list[dict[str, object]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_metric(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return float(statistics.median(
        run_mean(run, indices, getter) for run in runs
    ))


def tick_us(record: dict[str, object], field: str) -> float:
    return float(record[field]) / base.TICKS_PER_US


def rows(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
) -> dict[str, float]:
    return {
        name: median_metric(
            runs, indices,
            lambda record, name=name: base.generation_row_us(record, name),
        )
        for name in ROWS
    }


def median_step_series(
    runs: list[list[dict[str, object]]],
    getter: Callable[[dict[str, object]], float],
) -> list[float]:
    return [
        float(statistics.median(getter(run[index]) for run in runs))
        for index in range(1, STEPS)
    ]


def linear_slope(values: list[float]) -> float:
    xs = list(range(64, 256))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, values)
    )
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator


def physical_and_correct(runs: list[list[dict[str, object]]]) -> dict[str, bool]:
    first16_exact = True
    physical = True
    cache_lifecycle = True
    sequences: list[tuple[int, ...]] = []
    for run in runs:
        tokens: list[int] = []
        codes: list[int] = []
        for index, profile in enumerate(run):
            step = profile["_step_record"]
            tokens.append(int(step["selected_token_id"]))
            codes.append(int(step["selected_logit_half_bits"]))
            expected_before = 0 if index == 0 else 63 + index
            expected_after = 64 if index == 0 else 64 + index
            physical &= bool(
                int(step["rpc_result"]) == 0
                and bool(step["pass"])
                and profile["backend"] == "standalone_fastrpc_dsp"
                and profile["qnn"] == "none"
                and int(profile["block_invocation_count"]) == LAYERS
                and int(profile["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                and int(profile["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                and int(profile["intermediate_ddr_read_bytes"]) == 0
                and int(profile["intermediate_ddr_write_bytes"]) == 0
                and int(profile["intermediate_spill_fill_count"]) == 0
                and int(profile["boundary_ddr_write_bytes"]) == 0
                and int(profile["ledger_unattributed_ticks"]) == 0
                and int(profile["generation_lm_head_batch_n_tiles"]) == 8
                and int(profile["generation_lm_head_command_count"]) == 594
                and int(profile["generation_lm_head_scale_resident_bytes"])
                    == 1_215_488
            )
            cache_lifecycle &= bool(
                int(profile["first_position"]) == expected_before
                and int(profile["valid_length"]) == expected_after
            )
            for layer_index in range(LAYERS):
                layer = profile[f"slice_layer_{layer_index}"]
                cache_lifecycle &= bool(
                    int(layer["layer_index"]) == layer_index
                    and int(layer["cache_valid_before"]) == expected_before
                    and int(layer["cache_valid_after"]) == expected_after
                )
                physical &= bool(
                    int(layer["hidden_ddr_read_bytes"]) == 0
                    and int(layer["hidden_ddr_write_bytes"]) == 0
                    and int(layer["layer_unattributed_ticks"]) == 0
                )
        first16_exact &= (
            tokens[:16] == EXPECTED_TOKENS and codes[:16] == EXPECTED_CODES
        )
        sequences.append(tuple(tokens))
    return {
        "first16_exact_EXP0168_tokens_and_codes": first16_exact,
        "all_timed_physical_contracts": physical,
        "all_layer_cache_lengths_monotonic_and_synchronous": cache_lifecycle,
        "ten_sessions_identical_token_sequence": len(set(sequences)) == 1,
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    runs = load_runs(result_dir)
    audit = json.loads(
        (result_dir / "audit/independent_reference.json").read_text()
    )
    gates = physical_and_correct(runs)
    gates["independent_integer_head_reference_193_of_193"] = bool(
        audit.get("summary", {}).get("implementation_gate") == "pass"
        and int(audit.get("summary", {}).get("verified_steps", -1)) == STEPS
        and int(audit.get("summary", {}).get("token_and_code_matches", -1))
            == STEPS
    )
    gates["EXP0163_transformer_and_cache_regression"] = (
        exp167.validate_exp0163_regression(
            result_dir / "w4u8_exp0163_regression.log"
        )
    )

    prefill = rows(runs, (0,))
    full_decode_indices = tuple(range(1, STEPS))
    full_decode = rows(runs, full_decode_indices)
    bucket_summary: dict[str, dict[str, object]] = {}
    for label, indices in BUCKETS:
        module_rows = rows(runs, indices)
        attention = {
            field.replace("_ticks", "_us"): median_metric(
                runs, indices,
                lambda record, field=field: tick_us(record, field),
            )
            for field in ATTENTION_TICK_FIELDS
        }
        counters = {
            field: median_metric(
                runs, indices,
                lambda record, field=field: float(record[field]),
            )
            for field in CACHE_COUNTER_FIELDS
        }
        bucket_summary[label] = {
            "indices": [indices[0], indices[-1]],
            "modules_us": module_rows,
            "attention_us": attention,
            "cache_counters_per_token": counters,
            "decode_tokens_per_second":
                1e6 / module_rows["Complete Host wall"],
        }

    wall_series = median_step_series(runs, base.host_wall_us)
    attention_series = median_step_series(
        runs, lambda record: base.generation_row_us(record, "QK-Softmax-AV")
    )
    cache_read_series = median_step_series(
        runs, lambda record: float(record["scan_cache_ddr_read_bytes"])
    )
    slopes = {
        "complete_host_wall_us_per_added_cache_token": linear_slope(wall_series),
        "QK_softmax_AV_us_per_added_cache_token": linear_slope(attention_series),
        "cache_DDR_read_bytes_per_added_cache_token":
            linear_slope(cache_read_series),
    }

    seal_steps = (32, 64, 96, 128, 160, 192)
    seal_events = []
    for index in seal_steps:
        wall = float(statistics.median(
            base.host_wall_us(run[index]) for run in runs
        ))
        before = float(statistics.median(
            base.host_wall_us(run[index - 1]) for run in runs
        ))
        neighbor_values = [before]
        if index + 1 < STEPS:
            neighbor_values.append(float(statistics.median(
                base.host_wall_us(run[index + 1]) for run in runs
            )))
        neighbor = statistics.mean(neighbor_values)
        seal_events.append({
            "cache_length_before": 63 + index,
            "valid_length_after": 64 + index,
            "host_wall_us": wall,
            "neighbor_mean_us": neighbor,
            "increment_us": wall - neighbor,
            "segment_seal_count": int(statistics.median(
                int(run[index]["u8_cache_segment_seal_count"])
                for run in runs
            )),
            "segment_sealed_bytes": int(statistics.median(
                int(run[index]["u8_cache_segment_sealed_bytes"])
                for run in runs
            )),
        })

    session1_decode = run_mean(runs[0], full_decode_indices, base.host_wall_us)
    session10_decode = full_decode["Complete Host wall"]
    all_gates = all(gates.values())
    gates["all_required"] = all_gates
    summary = {
        "experiment": "EXP-0169",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "sessions": 10,
        "prefill_tokens": 64,
        "continuous_decode_tokens_per_session": 192,
        "gates": gates,
        "direct": {
            "prefill_host_us": prefill["Complete Host wall"],
            "prefill_tokens_per_second": 64e6 / prefill["Complete Host wall"],
            "decode_192_host_us_per_token": full_decode["Complete Host wall"],
            "decode_192_tokens_per_second":
                1e6 / full_decode["Complete Host wall"],
            "session1_decode_host_us_per_token": session1_decode,
            "ten_session_median_decode_host_us_per_token": session10_decode,
        },
        "prefill_modules_us": prefill,
        "all_decode_modules_us": full_decode,
        "buckets": bucket_summary,
        "linear_slopes": slopes,
        "segment_seal_events": seal_events,
        "generated_token_ids": [
            int(item["_step_record"]["selected_token_id"])
            for item in runs[0]
        ],
        "quality_gate": "disabled_by_contract",
        "provenance": {
            "audit_reference": str(
                result_dir / "audit/independent_reference.json"
            ),
            "audit_reference_sha256": sha256_file(
                result_dir / "audit/independent_reference.json"
            ),
            "regression_sha256": sha256_file(
                result_dir / "w4u8_exp0163_regression.log"
            ),
            "logs": {
                path.name: sha256_file(path)
                for path in sorted((result_dir / "raw").glob("*.log"))
            },
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    def cell(value: float, wall: float) -> str:
        return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"

    lines = [
        "# EXP-0169 W4U8 long continuous-decode attribution", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Scope | Tokens | Host wall/token | Direct throughput |",
        "|---|---:|---:|---:|",
        f"| M64 prefill | 64 | {prefill['Complete Host wall']:.3f} us | {64e6/prefill['Complete Host wall']:.3f} tok/s |",
        f"| Continuous decode L64-L255 | 192 | {full_decode['Complete Host wall']:.3f} us | {1e6/full_decode['Complete Host wall']:.3f} tok/s |",
        f"| Session 1 decode diagnostic | 192 | {session1_decode:.3f} us | {1e6/session1_decode:.3f} tok/s |",
        f"| Ten-session median decode | 192 | {session10_decode:.3f} us | {1e6/session10_decode:.3f} tok/s |",
        "", "## Cache-length buckets", "",
        "| Cache length before step | Host wall/token | tok/s | QK-Softmax-AV | Cache DDR read/token | Cache DDR write/token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, _ in BUCKETS:
        entry = bucket_summary[label]
        modules = entry["modules_us"]
        counters = entry["cache_counters_per_token"]
        lines.append(
            f"| {label} | {modules['Complete Host wall']:.3f} us | "
            f"{entry['decode_tokens_per_second']:.3f} | "
            f"{modules['QK-Softmax-AV']:.3f} us | "
            f"{counters['scan_cache_ddr_read_bytes']:.0f} B | "
            f"{counters['scan_cache_ddr_write_bytes']:.0f} B |"
        )

    first = bucket_summary[BUCKETS[0][0]]["modules_us"]
    last = bucket_summary[BUCKETS[-1][0]]["modules_us"]
    lines += [
        "", "## Additive module ledger: first versus last bucket", "",
        "| Module | L64-95 | L224-255 | Change |",
        "|---|---:|---:|---:|",
    ]
    for name in ROWS:
        lines.append(
            f"| {name} | {cell(first[name], first['Complete Host wall'])} | "
            f"{cell(last[name], last['Complete Host wall'])} | "
            f"{last[name] - first[name]:+.3f} us |"
        )

    lines += [
        "", "## Attention and cache substage diagnostics", "",
        "| Diagnostic | L64-95 | L224-255 | Change |",
        "|---|---:|---:|---:|",
    ]
    first_attention = bucket_summary[BUCKETS[0][0]]["attention_us"]
    last_attention = bucket_summary[BUCKETS[-1][0]]["attention_us"]
    for name in first_attention:
        lines.append(
            f"| {name} | {first_attention[name]:.3f} us | "
            f"{last_attention[name]:.3f} us | "
            f"{last_attention[name] - first_attention[name]:+.3f} us |"
        )
    lines += [
        "", "## Linear slopes", "",
        "| Quantity | Slope per added cache token |",
        "|---|---:|",
        f"| Complete Host wall | {slopes['complete_host_wall_us_per_added_cache_token']:.6f} us |",
        f"| QK-Softmax-AV | {slopes['QK_softmax_AV_us_per_added_cache_token']:.6f} us |",
        f"| Cache DDR read | {slopes['cache_DDR_read_bytes_per_added_cache_token']:.3f} B |",
        "", "## Segment-seal events", "",
        "| Cache L before | Valid L after | Host wall | Delta vs neighbors | Seals | Sealed bytes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for event in seal_events:
        lines.append(
            f"| {event['cache_length_before']} | {event['valid_length_after']} | "
            f"{event['host_wall_us']:.3f} us | {event['increment_us']:+.3f} us | "
            f"{event['segment_seal_count']} | {event['segment_sealed_bytes']} |"
        )
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "EXP-0169 changes only generation capacity, audit and profiling. "
        "All timed DSP kernels and the EXP-0168 physical plan are unchanged. "
        "Semantic quality remains intentionally disabled.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "prefill_tokens_per_second": 64e6 / prefill["Complete Host wall"],
        "decode_192_tokens_per_second": 1e6 / full_decode["Complete Host wall"],
        "first_bucket_tokens_per_second":
            bucket_summary[BUCKETS[0][0]]["decode_tokens_per_second"],
        "last_bucket_tokens_per_second":
            bucket_summary[BUCKETS[-1][0]]["decode_tokens_per_second"],
        "slopes": slopes,
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0169 required gate failed")


if __name__ == "__main__":
    main()
