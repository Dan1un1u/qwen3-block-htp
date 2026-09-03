#!/usr/bin/env python3
"""Validate and summarize EXP-0168 W4U8 LM-head scheduling."""

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


STEPS = 16
LAYERS = 28
EXPECTED_TOKENS = [
    124491, 51272, 51272, 9092, 51272, 128014, 23186, 85301,
    23186, 23186, 23186, 23186, 105260, 37440, 23186, 5205,
]
EXPECTED_CODES = [
    159, 162, 160, 161, 161, 160, 162, 164,
    163, 165, 165, 160, 156, 156, 165, 158,
]
W4F16_SUMMARY = Path(
    "/mnt/d/llm_exp/results/qwen3-block-htp/exp0166/"
    "20260903T_exp0166_8e0dcf8_formal/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-branch",
        default="codex/exp-0168-w4u8-lm-head-batch8-overlap",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(
    result_dir: Path, label: str, mode: int, experiment: int,
) -> list[list[dict[str, object]]]:
    paths = sorted((result_dir / "raw").glob(f"{label}_??.log"))
    if len(paths) != 10:
        raise ValueError(f"expected ten {label} logs, got {len(paths)}")
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
            raise ValueError(f"incomplete {label} generation run: {path}")
        if int(finals[0].get("generation_mode", -1)) != mode:
            raise ValueError(f"wrong {label} generation mode: {path}")
        for index, (step, profile) in enumerate(zip(steps, profiles)):
            if (
                int(step.get("experiment", -1)) != experiment
                or int(profile.get("experiment", -1)) != experiment
                or int(step.get("generation_step", -1)) != index
                or int(profile.get("generation_step", -1)) != index
                or int(step.get("generation_mode", -1)) != mode
            ):
                raise ValueError(f"{label} identity mismatch: {path}:{index}")
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


def rows(
    runs: list[list[dict[str, object]]], indices: tuple[int, ...],
    names: tuple[str, ...],
) -> dict[str, float]:
    return {
        name: median_metric(
            runs, indices,
            lambda record, name=name: base.generation_row_us(record, name),
        )
        for name in names
    }


def physical_and_correct(
    runs: list[list[dict[str, object]]], mode: int,
    batch: int, commands: int, resident: int,
) -> bool:
    for run in runs:
        tokens = []
        codes = []
        for profile in run:
            step = profile["_step_record"]
            tokens.append(int(step["selected_token_id"]))
            codes.append(int(step["selected_logit_half_bits"]))
            if not (
                int(step["generation_mode"]) == mode
                and int(step["rpc_result"]) == 0
                and bool(step["pass"])
                and step.get("selected_logit_encoding") == "u8_code"
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
                and int(profile["generation_lm_head_batch_n_tiles"]) == batch
                and int(profile["generation_lm_head_command_count"]) == commands
                and int(profile["generation_lm_head_scale_resident_bytes"]) == resident
            ):
                return False
            for layer_index in range(LAYERS):
                layer = profile[f"slice_layer_{layer_index}"]
                if not (
                    int(layer["layer_index"]) == layer_index
                    and int(layer["hidden_ddr_read_bytes"]) == 0
                    and int(layer["hidden_ddr_write_bytes"]) == 0
                    and int(layer["layer_unattributed_ticks"]) == 0
                ):
                    return False
        if tokens != EXPECTED_TOKENS or codes != EXPECTED_CODES:
            return False
    return True


def speed(control_us: float, candidate_us: float) -> float:
    return (control_us / candidate_us - 1.0) * 100.0


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    control = load_runs(result_dir, "control", 8, 167)
    candidate = load_runs(result_dir, "candidate", 9, 168)
    audit = json.loads(
        (result_dir / "audit" / "independent_reference.json").read_text()
    )
    audit_tokens = [int(item["device_token"]) for item in audit["steps"]]
    audit_codes = [int(item["device_logit_code"]) for item in audit["steps"]]

    prefill = (0,)
    decode = tuple(range(1, STEPS))
    names = (
        "Token embedding",
        *(name for name, _ in base.BASE_LEDGER[:-2]),
        "Final model RMSNorm",
        "Streaming W4 LM head + greedy argmax",
        base.BASE_LEDGER[-2][0], base.BASE_LEDGER[-1][0],
        "True Host-DSP boundary", "Complete Host wall",
    )
    direct: dict[str, object] = {}
    for scope, indices in (("prefill", prefill), ("decode", decode)):
        direct[scope] = {
            "control": rows(control, indices, names),
            "candidate": rows(candidate, indices, names),
        }

    def session_us(run: list[dict[str, object]]) -> float:
        return sum(base.host_wall_us(item) for item in run)

    control_prefill = direct["prefill"]["control"]["Complete Host wall"]
    candidate_prefill = direct["prefill"]["candidate"]["Complete Host wall"]
    control_decode = direct["decode"]["control"]["Complete Host wall"]
    candidate_decode = direct["decode"]["candidate"]["Complete Host wall"]
    control_session = float(statistics.median(map(session_us, control)))
    candidate_session = float(statistics.median(map(session_us, candidate)))
    paired = {
        "prefill_speed_percent": float(statistics.median(
            speed(base.host_wall_us(c[0]), base.host_wall_us(n[0]))
            for c, n in zip(control, candidate)
        )),
        "decode_speed_percent": float(statistics.median(
            speed(
                run_mean(c, decode, base.host_wall_us),
                run_mean(n, decode, base.host_wall_us),
            ) for c, n in zip(control, candidate)
        )),
        "session_speed_percent": float(statistics.median(
            speed(session_us(c), session_us(n))
            for c, n in zip(control, candidate)
        )),
    }
    gates = {
        "control_physical_and_exact": physical_and_correct(
            control, 8, 4, 1187, 0
        ),
        "candidate_physical_and_exact": physical_and_correct(
            candidate, 9, 8, 594, 1215488
        ),
        "independent_integer_reference": (
            audit.get("summary", {}).get("implementation_gate") == "pass"
            and audit_tokens == EXPECTED_TOKENS
            and audit_codes == EXPECTED_CODES
        ),
        "exp0163_transformer_and_cache_regression":
            exp167.validate_exp0163_regression(
                result_dir / "w4u8_exp0163_regression.log"
            ),
        "paired_prefill_strict_improvement":
            paired["prefill_speed_percent"] > 0.0,
        "paired_decode_strict_improvement":
            paired["decode_speed_percent"] > 0.0,
        "paired_session_strict_improvement":
            paired["session_speed_percent"] > 0.0,
        "quality_gate_disabled": True,
    }
    gates["all_required"] = all(gates.values())

    diagnostic_fields = (
        "generation_lm_head_ticks",
        "generation_lm_head_weight_dma_ticks",
        "generation_lm_head_weight_dma_wait_ticks",
        "generation_lm_head_scale_dma_ticks",
        "generation_lm_head_expand_ticks",
        "generation_lm_head_hmx_ticks",
        "generation_lm_head_hmx_tail_wait_ticks",
        "generation_lm_head_argmax_ticks",
    )
    diagnostics = {}
    for field in diagnostic_fields:
        diagnostics[field.replace("_ticks", "_us")] = {
            "control": median_metric(
                control, decode,
                lambda record, field=field:
                    float(record[field]) / base.TICKS_PER_US,
            ),
            "candidate": median_metric(
                candidate, decode,
                lambda record, field=field:
                    float(record[field]) / base.TICKS_PER_US,
            ),
        }

    summary = {
        "experiment": "EXP-0168",
        "source_branch": args.source_branch,
        "source_commit": args.source_commit,
        "formal_evidence": str(result_dir),
        "gates": gates,
        "paired": paired,
        "direct": direct,
        "session": {
            "control_us": control_session,
            "candidate_us": candidate_session,
            "control_prefill_tok_s": 64e6 / control_prefill,
            "candidate_prefill_tok_s": 64e6 / candidate_prefill,
            "control_decode_tok_s": 1e6 / control_decode,
            "candidate_decode_tok_s": 1e6 / candidate_decode,
        },
        "lm_head_diagnostics": diagnostics,
        "generated_token_ids": EXPECTED_TOKENS,
        "generated_u8_max_codes": EXPECTED_CODES,
        "quality_gate": "disabled_by_contract",
        "provenance": {
            "audit_reference": str(
                result_dir / "audit" / "independent_reference.json"
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
    if W4F16_SUMMARY.exists():
        w4f16 = json.loads(W4F16_SUMMARY.read_text())
        w4f16_prefill = float(
            w4f16["direct"]["prefill"]["candidate_repeat_ten"]
                 ["Complete Host wall"]
        )
        w4f16_decode = float(
            w4f16["direct"]["decode"]["candidate_repeat_ten"]
                 ["Complete Host wall"]
        )
        summary["w4f16_structural_target"] = {
            "prefill_us": w4f16_prefill,
            "decode_us": w4f16_decode,
            "candidate_prefill_speed_percent":
                speed(w4f16_prefill, candidate_prefill),
            "candidate_decode_speed_percent":
                speed(w4f16_decode, candidate_decode),
        }

    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )

    def cell(value: float, wall: float) -> str:
        return f"{value:.3f} us ({100.0 * value / wall:.1f}%)"

    lines = [
        "# EXP-0168 W4U8 batch-eight phase-overlaid LM head", "",
        f"Source: `{args.source_branch}` @ `{args.source_commit}`", "",
        "## Direct full-stack result", "",
        "| Path | M64 prefill wall | Prefill tok/s | Decode wall/token | Decode tok/s | Complete 16-step wall |",
        "|---|---:|---:|---:|---:|---:|",
        f"| EXP-0167-compatible mode 8 | {control_prefill:.3f} us | {64e6/control_prefill:.3f} | {control_decode:.3f} us | {1e6/control_decode:.3f} | {control_session:.3f} us |",
        f"| EXP-0168 mode 9 | {candidate_prefill:.3f} us | {64e6/candidate_prefill:.3f} | {candidate_decode:.3f} us | {1e6/candidate_decode:.3f} | {candidate_session:.3f} us |",
        f"| Paired median speed | {paired['prefill_speed_percent']:+.3f}% | {paired['prefill_speed_percent']:+.3f}% | {paired['decode_speed_percent']:+.3f}% | {paired['decode_speed_percent']:+.3f}% | {paired['session_speed_percent']:+.3f}% |",
        "",
    ]
    for scope, title in (("prefill", "M64 prefill"), ("decode", "Decode token")):
        cwall = direct[scope]["control"]["Complete Host wall"]
        nwall = direct[scope]["candidate"]["Complete Host wall"]
        lines += [
            f"## {title} additive ledger", "",
            "| Module | Mode 8 control | Mode 9 candidate |",
            "|---|---:|---:|",
        ]
        for name in names:
            lines.append(
                f"| {name} | {cell(direct[scope]['control'][name], cwall)} | "
                f"{cell(direct[scope]['candidate'][name], nwall)} |"
            )
        lines.append("")
    lines += [
        "## LM-head diagnostics (decode median)", "",
        "| Diagnostic | Mode 8 | Mode 9 |",
        "|---|---:|---:|",
    ]
    for name, values in diagnostics.items():
        lines.append(
            f"| {name} | {values['control']:.3f} us | "
            f"{values['candidate']:.3f} us |"
        )
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---:|"]
    for name, passed in gates.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "", "Mode 9 uses batch-eight integer HMX, a VTCM-resident full "
        "bias/requant carrier, asynchronous next-group weight DMA, and "
        "parallel W4-to-S8 HVX expansion. Full logits never leave VTCM. "
        "Semantic quality is intentionally not gated.", "",
    ]
    (result_dir / "full_profiling_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "all_required_gates_pass": gates["all_required"],
        "control_prefill_tok_s": 64e6 / control_prefill,
        "candidate_prefill_tok_s": 64e6 / candidate_prefill,
        "control_decode_tok_s": 1e6 / control_decode,
        "candidate_decode_tok_s": 1e6 / candidate_decode,
        "paired": paired,
        "result_dir": str(result_dir),
    }, indent=2))
    if not gates["all_required"]:
        raise SystemExit("EXP-0168 required gate failed")


if __name__ == "__main__":
    main()
