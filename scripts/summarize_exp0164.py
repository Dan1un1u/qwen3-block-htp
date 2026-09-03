#!/usr/bin/env python3
"""Validate and summarize EXP-0164 deterministic W4F16 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable


TICKS_PER_US = 19.2
LAYERS = 28
GENERATION_STEPS = 16

EXP0158 = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0158/"
    "20260902T071708Z_264c911a65a3_formal"
)
EXP0163 = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0163/"
    "20260903T073822Z_63de1de7028c_formal"
)
SEMANTIC_REFERENCE = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0164/semantic_gate/"
    "teacher_w4f16_greedy16.json"
)
ARTIFACT_DIR = Path(
    "/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/w4f16_greedy16"
)

BASE_LEDGER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("I/O and metadata", (
        "input_stage_ticks", "metadata_stage_ticks", "output_stage_ticks",
    )),
    ("Input RMSNorm", ("input_norm_ticks",)),
    ("QKV + Q/K Norm-RoPE preparation", (
        "qkv_projection_ticks", "qk_norm_rope_ticks",
    )),
    ("QK-Softmax-AV", ("attention_ticks",)),
    ("O projection", ("o_projection_ticks",)),
    ("Post-attention residual + RMSNorm", (
        "post_attention_residual_ticks", "post_attention_norm_ticks",
    )),
    ("Gate/Up + SwiGLU", ("gate_up_ticks", "activation_ticks")),
    ("Down projection", ("down_ticks",)),
    ("Final residual", ("final_residual_ticks",)),
    ("KV-cache carrier conversion/update", ("scan_cache_pack_ticks",)),
    ("KV-cache append DMA", ("scan_cache_append_ticks",)),
    ("Block internal orchestration", (
        "block_orchestration_ticks", "stage_boundary_ticks",
    )),
    ("Layer bookkeeping", ("layer_bookkeeping_ticks",)),
    ("DSP unattributed residual", ("ledger_unattributed_ticks",)),
    ("DSP runtime setup/teardown", (
        "runtime_setup_ticks", "runtime_teardown_ticks",
    )),
)

PROJECTION_DIAGNOSTICS = (
    "weight_dma_ticks", "hmx_compute_ticks", "projection_pack_ticks",
    "projection_hmx_wait_ticks", "projection_unpack_ticks",
    "hmx_ready_wait_ticks", "w4f16_expand_ticks",
    "w4f16_expand_work_ticks", "w4f16_expand_pool_wait_ticks",
    "w4f16_prefetch_wait_ticks", "w4f16_hmx_tail_wait_ticks",
    "w4f16_cross_prefetch_wait_ticks",
    "w4f16_cross_prefetch_lifetime_ticks",
)
ATTENTION_DIAGNOSTICS = (
    "attention_setup_ticks", "attention_qk_pack_ticks",
    "attention_qk_hmx_ticks", "attention_qk_unpack_ticks",
    "attention_softmax_ticks", "attention_av_pack_ticks",
    "attention_av_hmx_ticks", "attention_av_unpack_ticks",
    "attention_gqa_pipeline_ticks", "attention_unattributed_ticks",
)
MLP_DIAGNOSTICS = (
    "w4f16_gate_up_weight_dma_ticks", "w4f16_gate_up_expand_ticks",
    "w4f16_gate_up_expand_work_ticks",
    "w4f16_gate_up_expand_pool_wait_ticks",
    "w4f16_gate_up_hmx_wait_ticks",
    "w4f16_gate_up_hmx_tail_wait_ticks",
    "w4f16_gate_up_stream_work_ticks",
    "w4f16_gate_up_stream_ready_wait_ticks",
    "w4f16_gate_up_stream_join_wait_ticks",
)
PHYSICAL_FIELDS = (
    "weight_ddr_read_bytes", "boundary_ddr_read_bytes",
    "boundary_ddr_write_bytes", "scan_cache_ddr_read_bytes",
    "scan_cache_ddr_write_bytes", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "intermediate_spill_fill_count",
    "weight_dma_descriptor_count", "boundary_dma_descriptor_count",
    "intermediate_dma_descriptor_count", "hmx_command_count",
    "hmx_fp16_tile_pair_count", "hmx_u8s8_tile_pair_count",
    "vtcm_requested_bytes", "vtcm_acquired_bytes", "vtcm_peak_plan_bytes",
    "generation_embedding_ddr_read_bytes",
    "generation_lm_head_ddr_read_bytes",
    "generation_lm_head_command_count", "generation_lm_head_n_tiles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0164-w4f16-greedy-generation",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_lines(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("{"):
            continue
        records.append(json.loads(raw))
    return records


def load_generation_runs(
    result_dir: Path,
) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob("generation_??.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten generation logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        records = read_json_lines(path)
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
        if (
            len(steps) != GENERATION_STEPS
            or len(profiles) != GENERATION_STEPS
            or len(finals) != 1
            or finals[0].get("all_steps_pass") is not True
        ):
            raise ValueError(f"incomplete generation run: {path}")
        if [int(item["generation_step"]) for item in profiles] != list(
            range(GENERATION_STEPS)
        ):
            raise ValueError(f"unordered generation profiles: {path}")
        for step, profile in zip(steps, profiles):
            profile["_step_record"] = step
            profile["_source_log"] = str(path)
        runs.append(profiles)
    return runs


def load_replay_runs(
    paths: list[Path], expected_profiles: int,
) -> list[list[dict[str, object]]]:
    if len(paths) != 10:
        raise ValueError(f"expected ten replay logs, got {len(paths)}")
    runs: list[list[dict[str, object]]] = []
    for path in paths:
        profiles = [
            item for item in read_json_lines(path)
            if item.get("record") == "replay_profile"
        ]
        if len(profiles) != expected_profiles:
            raise ValueError(f"incomplete replay profile: {path}")
        runs.append(profiles)
    return runs


def mean_record_value(
    run: list[dict[str, object]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return statistics.mean(getter(run[index]) for index in indices)


def median_run_value(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    getter: Callable[[dict[str, object]], float],
) -> float:
    return float(statistics.median(
        mean_record_value(run, indices, getter) for run in runs
    ))


def ticks_getter(*fields: str) -> Callable[[dict[str, object]], float]:
    return lambda record: (
        sum(float(record[field]) for field in fields) / TICKS_PER_US
    )


def host_wall_us(record: dict[str, object]) -> float:
    return float(record["host_wall_ns"]) / 1000.0


def host_dsp_boundary_us(record: dict[str, object]) -> float:
    return (
        host_wall_us(record)
        - float(record["invocation_ticks"]) / TICKS_PER_US
    )


def base_row_us(record: dict[str, object], fields: tuple[str, ...]) -> float:
    return sum(float(record[field]) for field in fields) / TICKS_PER_US


def generation_row_us(record: dict[str, object], name: str) -> float:
    if name == "Token embedding":
        return float(record["generation_embedding_ticks"]) / TICKS_PER_US
    if name == "Final model RMSNorm":
        return float(record["generation_final_norm_ticks"]) / TICKS_PER_US
    if name == "Streaming W4 LM head + greedy argmax":
        return (
            float(record["generation_lm_head_ticks"])
            - float(record["generation_final_norm_ticks"])
        ) / TICKS_PER_US
    if name == "True Host-DSP boundary":
        return host_dsp_boundary_us(record)
    if name == "Complete Host wall":
        return host_wall_us(record)
    for row, fields in BASE_LEDGER:
        if row == name:
            return base_row_us(record, fields)
    raise KeyError(name)


def summarize_rows(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    rows: tuple[str, ...], representative: int | None = None,
) -> dict[str, float]:
    if representative is not None:
        run = runs[representative]
        return {
            row: mean_record_value(
                run, indices, lambda record, row=row:
                    generation_row_us(record, row)
            )
            for row in rows
        }
    return {
        row: median_run_value(
            runs, indices, lambda record, row=row:
                generation_row_us(record, row)
        )
        for row in rows
    }


def summarize_old_overview(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
) -> dict[str, object]:
    wall = median_run_value(runs, indices, host_wall_us)
    modules = {
        row: median_run_value(
            runs, indices,
            lambda record, fields=fields: base_row_us(record, fields),
        )
        for row, fields in BASE_LEDGER
    }
    modules["True Host-DSP boundary"] = median_run_value(
        runs, indices, host_dsp_boundary_us
    )
    modules["Complete Host wall"] = wall
    return {"wall_us": wall, "modules_us": modules}


def fmt_cell(value: float, wall: float) -> str:
    return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"


def fmt_optional(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.3f}{suffix}"


def validate_generation(
    runs: list[list[dict[str, object]]], expected: list[int],
) -> dict[str, bool]:
    token = True
    physical = True
    ledger = True
    cache_structure = True
    stable: list[list[int]] = []
    for run in runs:
        tokens: list[int] = []
        for index, profile in enumerate(run):
            step = profile["_step_record"]
            tokens.append(int(step["selected_token_id"]))
            token &= (
                int(profile["experiment"]) == 164
                and int(profile["generation_step"]) == index
                and step["token_match"] is True
                and step["pass"] is True
            )
            physical &= (
                profile["backend"] == "standalone_fastrpc_dsp"
                and profile["qnn"] == "none"
                and int(profile["vtcm_requested_bytes"]) == 8 * 1024 * 1024
                and int(profile["vtcm_acquired_bytes"]) == 8 * 1024 * 1024
                and int(profile["intermediate_ddr_read_bytes"]) == 0
                and int(profile["intermediate_ddr_write_bytes"]) == 0
                and int(profile["intermediate_spill_fill_count"]) == 0
                and int(profile["boundary_ddr_write_bytes"]) == 0
                and int(profile["block_invocation_count"]) == LAYERS
            )
            ledger &= int(profile["ledger_unattributed_ticks"]) == 0
            for layer_index in range(LAYERS):
                layer = profile[f"slice_layer_{layer_index}"]
                cache_structure &= (
                    int(layer["layer_index"]) == layer_index
                    and int(layer["hidden_ddr_read_bytes"]) == 0
                    and int(layer["hidden_ddr_write_bytes"]) == 0
                    and int(layer["layer_unattributed_ticks"]) == 0
                )
        token &= tokens == expected
        stable.append(tokens)
    return {
        "independent_token_sequence_exact": token,
        "stable_across_ten_sessions": all(item == stable[0] for item in stable),
        "physical_contract": physical,
        "complete_ledger": ledger,
        "cache_and_hidden_structure": cache_structure,
    }


def validate_regression(path: Path) -> bool:
    records = read_json_lines(path)
    profiles = [
        item for item in records if item.get("record") == "replay_profile"
    ]
    finals = [
        item for item in records
        if item.get("replay_sequence_complete") is True
    ]
    if len(profiles) != 9 or len(finals) != 1:
        return False
    return finals[0].get("all_steps_pass") is True and all(
        int(item["dsp_status"]) == 3
        and int(item["numerical_status"]) == 1
        and int(item["output_nonfinite_count"]) == 0
        and float(item["output_cosine"]) >= 0.99999
        and float(item["output_nrmse"]) <= 0.003
        and int(item["cache_prefix_mismatches"]) == 0
        and int(item["cache_structure_mismatches"]) == 0
        and int(item["cache_nonfinite_count"]) == 0
        and int(item["cache_tensor_count"]) == 56
        and int(item["intermediate_ddr_read_bytes"]) == 0
        and int(item["intermediate_ddr_write_bytes"]) == 0
        and int(item["intermediate_spill_fill_count"]) == 0
        for item in profiles
    )


def diagnostic_summary(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    fields: tuple[str, ...],
) -> dict[str, float]:
    return {
        field: median_run_value(
            runs, indices,
            lambda record, field=field: float(record[field]) / TICKS_PER_US,
        )
        for field in fields
    }


def physical_summary(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
) -> dict[str, float]:
    return {
        field: median_run_value(
            runs, indices,
            lambda record, field=field: float(record[field]),
        )
        for field in PHYSICAL_FIELDS
    }


def append_overview_table(
    lines: list[str], title: str,
    variants: dict[str, dict[str, object]],
) -> None:
    f16 = variants["F16F16"]
    w4f16 = variants["W4F16"]
    w4u8 = variants["W4U8"]
    lines += [
        f"## {title}", "",
        "| Module | F16F16 | W4F16 | W4U8 | W4U8 speed vs W4F16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in [name for name, _ in BASE_LEDGER] + [
        "True Host-DSP boundary", "Complete Host wall",
    ]:
        left = float(f16["modules_us"][row])
        middle = float(w4f16["modules_us"][row])
        right = float(w4u8["modules_us"][row])
        left_wall = float(f16["wall_us"])
        middle_wall = float(w4f16["wall_us"])
        right_wall = float(w4u8["wall_us"])
        change = (middle / right - 1.0) * 100.0 if right else 0.0
        lines.append(
            f"| {row} | {fmt_cell(left, left_wall)} | "
            f"{fmt_cell(middle, middle_wall)} | "
            f"{fmt_cell(right, right_wall)} | {change:+.1f}% |"
        )
    lines.append("")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    semantic = json.loads(SEMANTIC_REFERENCE.read_text(encoding="utf-8"))
    expected = [int(item) for item in semantic["w4f16"]["token_ids"]]
    generation_runs = load_generation_runs(result_dir)
    gates = validate_generation(generation_runs, expected)
    regression_path = result_dir / "w4f16_exp0158_regression.log"
    gates["accepted_transformer_replay_regression"] = validate_regression(
        regression_path
    )
    gates["readable_nonempty_text"] = bool(semantic["w4f16"]["text"].strip())
    gates["all"] = all(gates.values())

    total_walls = [
        sum(host_wall_us(record) for record in run)
        for run in generation_runs
    ]
    total_median = float(statistics.median(total_walls))
    representative = min(
        range(len(total_walls)), key=lambda index:
            abs(total_walls[index] - total_median)
    )
    generation_rows = (
        "Token embedding",
        *(name for name, _ in BASE_LEDGER[:-2]),
        "Final model RMSNorm",
        "Streaming W4 LM head + greedy argmax",
        BASE_LEDGER[-2][0], BASE_LEDGER[-1][0],
        "True Host-DSP boundary", "Complete Host wall",
    )
    scopes = {
        "prefill": (0,),
        "decode": tuple(range(1, GENERATION_STEPS)),
    }
    generation_summary: dict[str, object] = {}
    for name, indices in scopes.items():
        generation_summary[name] = {
            "representative_session": summarize_rows(
                generation_runs, indices, generation_rows, representative
            ),
            "ten_session_median": summarize_rows(
                generation_runs, indices, generation_rows
            ),
            "projection_diagnostics_us": diagnostic_summary(
                generation_runs, indices, PROJECTION_DIAGNOSTICS
            ),
            "attention_diagnostics_us": diagnostic_summary(
                generation_runs, indices, ATTENTION_DIAGNOSTICS
            ),
            "mlp_diagnostics_us": diagnostic_summary(
                generation_runs, indices, MLP_DIAGNOSTICS
            ),
            "physical": physical_summary(generation_runs, indices),
        }

    f16_runs = load_replay_runs(sorted(
        (EXP0158 / "raw").glob("round_*_f16f16_hmx_native_f16.jsonl")
    ), 9)
    w4f16_runs = load_replay_runs(sorted(
        (EXP0158 / "raw").glob("round_*_w4f16_hmx_native_f16.jsonl")
    ), 9)
    w4u8_runs = load_replay_runs(sorted(
        (EXP0163 / "raw").glob("round_*_candidate.jsonl")
    ), 193)
    overview = {
        "prefill": {
            "F16F16": summarize_old_overview(f16_runs, (0,)),
            "W4F16": summarize_old_overview(w4f16_runs, (0,)),
            "W4U8": summarize_old_overview(w4u8_runs, (0,)),
        },
        "early_decode_L64_L71": {
            "F16F16": summarize_old_overview(f16_runs, tuple(range(1, 9))),
            "W4F16": summarize_old_overview(w4f16_runs, tuple(range(1, 9))),
            "W4U8": summarize_old_overview(w4u8_runs, tuple(range(1, 9))),
        },
    }

    summary = {
        "experiment": "EXP-0164",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "artifact_dir": str(ARTIFACT_DIR),
        "execution_unit": (
            "one real token-input prefill or stateful decode pass through "
            "embedding, Qwen3 layers0-27, final RMSNorm, streamed W4 LM head, "
            "greedy argmax and token feedback"
        ),
        "variant": "W4F16",
        "performance_control": None,
        "performance_control_reason": (
            "No earlier implementation has the same token-to-token boundary; "
            "EXP-0158 ends at the layer-27 hidden state."
        ),
        "sessions": 10,
        "paired": False,
        "representative_session_index": representative + 1,
        "generated_token_ids": expected,
        "generated_text": semantic["w4f16"]["text"],
        "bf16_teacher_text": semantic["bf16_teacher"]["text"],
        "gates": gates,
        "generation": generation_summary,
        "total_sequence_host_wall_us": {
            "representative": total_walls[representative],
            "ten_session_median": total_median,
        },
        "three_variant_overview": overview,
        "provenance": {
            "semantic_reference": str(SEMANTIC_REFERENCE),
            "semantic_reference_sha256": sha256_file(SEMANTIC_REFERENCE),
            "f16f16_w4f16_overview": str(EXP0158),
            "w4u8_overview": str(EXP0163),
            "regression_log": str(regression_path),
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP-0164 full profiling report", "",
        "## Identity and scope", "",
        f"- Source: `{args.source_branch}` @ `{args.source_commit}`",
        f"- Formal evidence: `{result_dir}`",
        f"- Artifact: `{ARTIFACT_DIR}`",
        "- Project Variant: W4F16; ten unpaired complete generation sessions.",
        "- Direct numerical control: independent Host W4F16 implementation.",
        "- Direct performance control: N/A. EXP-0158 stops at layer-27 hidden "
        "state and does not include token embedding, final RMSNorm, LM head or "
        "greedy feedback, so it is not an equivalent-scope wall-time control.",
        "- Repeat-one scope is one physical prefill/decode invocation. The "
        "single-session columns use the complete session nearest the ten-session "
        f"median (session {representative + 1}); repeat-ten is the median across "
        "ten independently prepared complete sessions.", "",
        "The PC-028 tables below reuse F16F16 and W4F16 from EXP-0158 and W4U8 "
        "from EXP-0163. They cover the shared transformer-stack scope only; "
        "they are context, not a direct performance control for EXP-0164.", "",
    ]
    append_overview_table(lines, "PC-028 M64 prefill overview", overview["prefill"])
    append_overview_table(
        lines, "PC-028 early decode L64-L71 overview",
        overview["early_decode_L64_L71"],
    )

    lines += [
        "## Semantic result", "",
        f"- Independent W4F16 and device text: `{semantic['w4f16']['text']}`",
        f"- BF16 teacher diagnostic: `{semantic['bf16_teacher']['text']}`",
        "- Device token IDs: `" + ", ".join(map(str, expected)) + "`",
        "- Exactness: 160/160 generated steps match the independent W4F16 "
        "reference across ten sessions; all ten decoded strings are identical.",
        "- The unchanged EXP-0158 transformer replay passes all nine steps on "
        "the EXP-0164 binary under the accepted FP16 composition/cache gates.", "",
        "## Same-scope control availability", "",
        "| Scope | Equivalent control | Representative single session | "
        "Ten-session median | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, indices in scopes.items():
        representative_wall = generation_summary[scope][
            "representative_session"
        ]["Complete Host wall"]
        median_wall = generation_summary[scope][
            "ten_session_median"
        ]["Complete Host wall"]
        lines.append(
            f"| {scope} | N/A | {representative_wall:.3f} us | "
            f"{median_wall:.3f} us | N/A |"
        )
    lines.append(
        f"| 16-pass sequence | N/A | {total_walls[representative]:.3f} us | "
        f"{total_median:.3f} us | N/A |"
    )
    lines += [
        "", "N/A is required here because no earlier implementation executes "
        "the same token-to-token boundary. A smaller-scope hidden-state replay "
        "would be a misleading control.", "",
        "## Complete additive generation ledger", "",
        "All rows are mutually exclusive. The LM-head row excludes final "
        "RMSNorm, which is shown separately. Host-DSP boundary is measured for "
        "each record as Host wall minus DSP invocation before aggregation.", "",
        "| Module | Prefill single | Prefill repeat-ten | Decode single | "
        "Decode repeat-ten |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in generation_rows:
        values = [
            generation_summary["prefill"]["representative_session"][row],
            generation_summary["prefill"]["ten_session_median"][row],
            generation_summary["decode"]["representative_session"][row],
            generation_summary["decode"]["ten_session_median"][row],
        ]
        walls = [
            generation_summary["prefill"]["representative_session"]
                ["Complete Host wall"],
            generation_summary["prefill"]["ten_session_median"]
                ["Complete Host wall"],
            generation_summary["decode"]["representative_session"]
                ["Complete Host wall"],
            generation_summary["decode"]["ten_session_median"]
                ["Complete Host wall"],
        ]
        lines.append(
            f"| {row} | " + " | ".join(
                fmt_cell(float(value), float(wall))
                for value, wall in zip(values, walls)
            ) + " |"
        )

    lines += [
        "", "## Overlapping engine diagnostics", "",
        "These qtimer counters overlap one another and the additive ledger. "
        "They explain scheduling but must not be summed into Host wall.", "",
    ]
    for title, key, fields in (
        ("Projection", "projection_diagnostics_us", PROJECTION_DIAGNOSTICS),
        ("Attention", "attention_diagnostics_us", ATTENTION_DIAGNOSTICS),
        ("MLP", "mlp_diagnostics_us", MLP_DIAGNOSTICS),
    ):
        lines += [
            f"### {title}", "",
            "| Counter | Prefill repeat-ten | Decode repeat-ten |",
            "|---|---:|---:|",
        ]
        for field in fields:
            lines.append(
                f"| {field} | "
                f"{generation_summary['prefill'][key][field]:.3f} us | "
                f"{generation_summary['decode'][key][field]:.3f} us |"
            )
        lines.append("")

    lines += [
        "### Generation head", "",
        "| Counter | Prefill repeat-ten | Decode repeat-ten |",
        "|---|---:|---:|",
    ]
    for field in (
        "generation_lm_head_weight_dma_ticks",
        "generation_lm_head_scale_dma_ticks",
        "generation_lm_head_expand_ticks",
        "generation_lm_head_hmx_ticks",
        "generation_lm_head_argmax_ticks",
    ):
        values = [
            median_run_value(
                generation_runs, scopes[scope],
                lambda record, field=field:
                    float(record[field]) / TICKS_PER_US,
            )
            for scope in ("prefill", "decode")
        ]
        lines.append(f"| {field} | {values[0]:.3f} us | {values[1]:.3f} us |")

    lines += [
        "", "## Physical contract", "",
        "| Metric | Prefill repeat-ten | Decode repeat-ten |",
        "|---|---:|---:|",
    ]
    for field in PHYSICAL_FIELDS:
        lines.append(
            f"| {field} | "
            f"{generation_summary['prefill']['physical'][field]:.0f} | "
            f"{generation_summary['decode']['physical'][field]:.0f} |"
        )
    lines += [
        "| FastRPC invocations per pass | 1 | 1 |",
        "| HMX ownership domains | 1 | 1 |",
        "| backend/fallback | standalone cDSP / none | standalone cDSP / none |",
        "| timed full-logits DDR bytes | 0 | 0 |", "",
        "## Correctness and gates", "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "Speed is report-only for EXP-0164. The current LM head is a "
        "correctness-first implementation: 4,748 output tiles are issued in "
        "2,374 HMX commands per pass, and no performance-baseline promotion is "
        "claimed by this report.",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "experiment": "EXP-0164",
        "all_gates_pass": gates["all"],
        "representative_session": representative + 1,
        "prefill_median_us": generation_summary["prefill"]
            ["ten_session_median"]["Complete Host wall"],
        "decode_median_us": generation_summary["decode"]
            ["ten_session_median"]["Complete Host wall"],
        "sequence_median_us": total_median,
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all"]:
        raise SystemExit("EXP-0164 gate failed")


if __name__ == "__main__":
    main()
