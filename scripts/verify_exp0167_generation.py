#!/usr/bin/env python3
"""Verify EXP-0167 final RMSNorm and W4U8 LM-head implementation."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from reference_w4u8_hmx import (
    HmxU8Converter,
    exact_rms_norm_u8,
    load_qparams_bin,
    project_w4u8,
    projection_bias_words,
    unpack_w4_codes,
)


VOCAB = 151_936
HIDDEN = 2_048
HMX_CHANNELS = 32
QPARAM_RECORD = struct.Struct("<32sfi2f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--experiment-record", type=int, default=167)
    parser.add_argument("--experiment-label", default="EXP-0167")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--package", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/"
            "w4u8_greedy16"
        ),
    )
    parser.add_argument(
        "--converter", type=Path,
        default=Path(
            "/home/daniuniu/work/qwen3-block-htp/build/reference/"
            "qbh_hmx_u8_reference.so"
        ),
    )
    parser.add_argument(
        "--transformer-package", type=Path, default=None,
        help="package containing layer27/qparams_u8.bin",
    )
    return parser.parse_args()


def load_generation_qparams(path: Path) -> dict[str, dict[str, object]]:
    payload = path.read_bytes()
    if len(payload) != 2 * QPARAM_RECORD.size:
        raise ValueError("generation qparam byte count mismatch")
    result: dict[str, dict[str, object]] = {}
    for raw_name, scale, zero_point, minimum, maximum in \
            QPARAM_RECORD.iter_unpack(payload):
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        result[name] = {
            "scale": scale, "zero_point": zero_point,
            "minimum": minimum, "maximum": maximum,
        }
    return result


def load_device_steps(
    path: Path, experiment_record: int, steps: int,
) -> list[dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (record.get("experiment") == experiment_record and
                "generation_step" in record and
                "selected_token_id" in record):
            result[int(record["generation_step"])] = record
    if sorted(result) != list(range(steps)):
        raise ValueError(f"missing device generation steps: {sorted(result)}")
    return [result[index] for index in range(steps)]


def audit_bias(package: Path, final_qparam: dict[str, object],
               output_qparam: dict[str, object]) -> int:
    weights = unpack_w4_codes(
        package, "generation_lm_head", VOCAB, HIDDEN
    )
    scales = np.fromfile(
        package / "generation_lm_head_weight_w4_scale_f32.bin",
        dtype="<f4",
    )
    lower, upper = projection_bias_words(
        weights, scales, final_qparam, output_qparam
    )
    expected = np.empty((VOCAB // HMX_CHANNELS, 2, HMX_CHANNELS),
                        dtype="<u4")
    expected[:, 0, :] = lower.reshape(-1, HMX_CHANNELS)
    expected[:, 1, :] = upper.reshape(-1, HMX_CHANNELS).view("<u4")
    actual = np.fromfile(
        package / "generation_lm_head_bias_u32.bin", dtype="<u4"
    ).reshape(expected.shape)
    return int(np.count_nonzero(actual != expected))


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    package = args.package.resolve()
    transformer_package = (
        args.transformer_package.resolve()
        if args.transformer_package is not None else package
    )
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device_steps = load_device_steps(
        audit_dir / "device.jsonl", args.experiment_record, args.steps
    )
    layer_qparams = load_qparams_bin(
        transformer_package / "layer27/qparams_u8.bin"
    )
    generation_qparams = load_generation_qparams(
        package / "generation_qparams_u8.bin"
    )
    final_qparam = generation_qparams["generation_final_norm_output"]
    output_qparam = generation_qparams["generation_lm_head_output"]
    gamma = np.fromfile(
        package / "generation_final_norm_weight_f16.bin", dtype="<f2"
    )
    converter = HmxU8Converter(args.converter)
    bias_mismatches = audit_bias(package, final_qparam, output_qparam)

    records: list[dict[str, object]] = []
    all_match = bias_mismatches == 0
    for step, device in enumerate(device_steps):
        hidden = np.fromfile(
            audit_dir / f"generation_hidden_step{step:02d}_u8.bin",
            dtype=np.uint8,
        )
        if hidden.shape != (HIDDEN,):
            raise ValueError(f"step {step} hidden shape {hidden.shape}")
        normalized = exact_rms_norm_u8(
            hidden.reshape(1, HIDDEN),
            layer_qparams["block_output"], gamma, final_qparam,
        )
        logits = project_w4u8(
            normalized, package, "generation_lm_head",
            VOCAB, HIDDEN, final_qparam, output_qparam, converter,
        )[0]
        token = int(np.argmax(logits))
        code = int(logits[token])
        match = (token == int(device["selected_token_id"]) and
                 code == int(device["selected_logit_half_bits"]))
        all_match &= match
        records.append({
            "step": step,
            "device_token": int(device["selected_token_id"]),
            "reference_token": token,
            "device_logit_code": int(device["selected_logit_half_bits"]),
            "reference_logit_code": code,
            "final_norm_saturation_count": int(
                np.count_nonzero((normalized == 0) | (normalized == 255))
            ),
            "match": match,
        })
        print(json.dumps(records[-1], sort_keys=True), flush=True)

    summary = {
        "experiment": args.experiment_label,
        "bias_carrier_mismatches": bias_mismatches,
        "verified_steps": len(records),
        "token_and_code_matches": sum(bool(x["match"]) for x in records),
        "teacher_quality_gate": "disabled",
        "implementation_gate": "pass" if all_match else "fail",
    }
    (audit_dir / "independent_reference.json").write_text(
        json.dumps({"summary": summary, "steps": records}, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if not all_match:
        raise SystemExit(f"{args.experiment_label} implementation gate failed")


if __name__ == "__main__":
    main()
