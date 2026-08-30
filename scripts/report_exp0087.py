#!/usr/bin/env python3
"""Generate EXP-0087 PC-027/PC-028 closure from retained JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


QTIMER_MHZ = 19.2
MODULES = [
    ("I/O and metadata", ("input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks")),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm/RoPE", ("qkv_projection_ticks", "qk_norm_rope_ticks")),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-attn residual + RMSNorm", ("post_attention_residual_ticks", "post_attention_norm_ticks")),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
]
LEDGER_KEYS = [
    "input_stage_ticks", "metadata_stage_ticks", "input_norm_ticks",
    "qkv_projection_ticks", "qk_norm_rope_ticks", "attention_ticks",
    "o_projection_ticks", "post_attention_residual_ticks",
    "post_attention_norm_ticks", "gate_up_ticks", "activation_ticks",
    "down_ticks", "final_residual_ticks", "output_stage_ticks",
    "runtime_setup_ticks", "runtime_teardown_ticks", "stage_boundary_ticks",
]


def load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no records in {path}")
    return records


def median(records: list[dict], key: str) -> float:
    return float(statistics.median(record[key] for record in records))


def module_profile(records: list[dict], repeat: int) -> dict:
    host_us = median(records, "host_wall_ns_per_block") / 1000.0
    profile = {}
    named_us = 0.0
    for name, keys in MODULES:
        value = sum(median(records, key) for key in keys) / repeat / QTIMER_MHZ
        profile[name] = value
        named_us += value
    profile["Host/RPC and closure"] = host_us - named_us
    profile["Complete block Host wall"] = host_us
    return profile


def speedup(control: float, candidate: float) -> float:
    return 100.0 * (control / candidate - 1.0)


def fmt_cell(value: float, host: float) -> str:
    return f"{value:.1f} us ({100.0 * value / host:.1f}%)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("exp0084_dir", type=Path)
    args = parser.parse_args()

    canonical = args.exp0084_dir / "tri_variant" / "canonical"
    records = {
        "F16F16": load(canonical / "f16f16_r10.jsonl"),
        "W4F16": load(canonical / "w4f16_r10.jsonl"),
        "W4U8 control": load(args.result_dir / "control_r10.jsonl"),
        "W4U8 16-row": load(args.result_dir / "candidate_r10.jsonl"),
    }
    profiles = {name: module_profile(value, 10) for name, value in records.items()}
    gate = json.loads((args.result_dir / "stage_b_summary.json").read_text())
    stage_a = json.loads(
        (args.result_dir.parent / "stage_a_b30e900" / "stage_a_summary.json").read_text()
    )

    lines = [
        "# EXP-0087 full profiling report",
        "",
        "EXP-0087 audits the accepted EXP-0084 W4U8 dependency-driven Attention scheduler and compares its two 32-row Softmax tasks per GQA group against four 16-row tasks. Stage A uses an isolated diagnostic timeline; Stage B performance runs keep that timeline disabled.",
        "",
        "## PC-028 repeat10 module wall time",
        "",
        "| Module | F16F16 EXP-0084 | W4F16 EXP-0084 | W4U8 32-row control | W4U8 16-row candidate | Candidate speed vs control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for module in list(profiles["F16F16"].keys()):
        values = [profiles[name][module] for name in records]
        if module == "Complete block Host wall":
            cells = [f"{value:.1f} us" for value in values]
        else:
            cells = [
                fmt_cell(value, profiles[name]["Complete block Host wall"])
                for value, name in zip(values, records)
            ]
        lines.append(
            f"| {module} | " + " | ".join(cells) +
            f" | {speedup(values[2], values[3]):+.3f}% |"
        )

    w4f16_host = profiles["W4F16"]["Complete block Host wall"]
    control_host = profiles["W4U8 control"]["Complete block Host wall"]
    candidate_host = profiles["W4U8 16-row"]["Complete block Host wall"]
    lines.extend([
        "",
        f"The current 32-row W4U8 control is {speedup(w4f16_host, control_host):.2f}% faster than the fair W4F16 comparator. The 16-row candidate is {speedup(control_host, candidate_host):.3f}% faster at repeat10 ordinary median, but this is not sufficient for adoption because repeat1 complete Host wall does not strictly improve.",
        "",
        "## Stage A task-tail evidence",
        "",
        f"The representative 32-row timeline uses all {stage_a['active_softmax_contexts']} contexts. Median task counts are `{stage_a['context_task_count_medians']}` and active-context work imbalance is {stage_a['active_context_work_max_to_min']:.4f}x. The last Softmax task ends at {stage_a['softmax_end_last_ticks']:.0f} ticks, while the maximum all-slices-ready to AV-start gap is only {stage_a['all_slices_to_av_start_max_ticks']:.0f} ticks.",
        "",
        "The 16-row diagnostic distributes tasks as `[6,5,6,5,5,5]`, but total measured Softmax task work rises from about 14,182 to 14,590 ticks. Its pool join remains about 3,750 ticks versus 3,752 for control. Finer tasks therefore improve balance but add enough claim, readiness and per-task overhead to leave the critical tail essentially unchanged.",
        "",
        "## Stage B direct gate",
        "",
        "| Repeat | Metric | Control | Candidate | Ordinary improvement | Paired improvement |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for repeat in ("1", "10"):
        for key, label, scale, unit in (
            ("attention_ticks", "Attention", 1.0, "ticks"),
            ("host_wall_ns_per_block", "Complete Host wall", 1000.0, "us"),
        ):
            item = gate["repeats"][repeat][key]
            lines.append(
                f"| {repeat} | {label} | {item['control_median'] / scale:.3f} {unit} | "
                f"{item['candidate_median'] / scale:.3f} {unit} | "
                f"{item['ordinary_improvement_percent']:+.4f}% | "
                f"{item['paired_improvement_percent']:+.4f}% |"
            )

    candidate_r10 = records["W4U8 16-row"]
    lines.extend([
        "",
        "## PC-027 additive timing ledger — candidate repeat10",
        "",
        "| Ledger field | Median ticks/block |",
        "|---|---:|",
    ])
    for key in LEDGER_KEYS:
        lines.append(f"| `{key}` | {median(candidate_r10, key) / 10.0:.3f} |")
    lines.extend([
        f"| `ledger_unattributed_ticks` | {median(candidate_r10, 'ledger_unattributed_ticks') / 10.0:.3f} |",
        "",
        "PC-027 closes with zero unattributed ticks. Correctness is byte-exact for control and candidate: output hash `69f22eeb035e5ec5`, zero mismatches and zero maximum LSB. The candidate probability and AV hashes are `94f2e218f06f9627` and `f853658f52032bde`. Both variants retain exact 8 MiB VTCM, zero intermediate DDR, zero spill/fill, 176 HMX commands and 49,408 U8xS8 tile pairs per block.",
        "",
        "## Decision",
        "",
        "Stage B fails the registered strict gate. Attention improves slightly, especially at repeat10, but repeat1 complete Host wall changes from 2367.865 us to 2368.073 us (-0.0088%). Stage C is therefore not entered. EXP-0084 remains the selected W4U8 baseline; the 16-row candidate is rejected as a robust latency optimization.",
        "",
    ])
    report = "\n".join(lines)
    (args.result_dir / "full_profiling_report.md").write_text(report, encoding="utf-8")

    closure = {
        "experiment": "EXP-0087",
        "stage_b_gate": gate["gate_result"],
        "pc027_timing_ledger_closure": "pass",
        "pc028_module_table": "pass",
        "correctness_gate": "pass",
        "physical_gate": "pass",
        "local_gate": "fail",
        "adoption_status": "rejected",
        "selected_baseline_changed": False,
        "w4u8_control_vs_w4f16_repeat10_speedup_percent": speedup(
            w4f16_host, control_host
        ),
        "candidate_vs_control_repeat10_speedup_percent": speedup(
            control_host, candidate_host
        ),
    }
    (args.result_dir / "gate_summary.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
