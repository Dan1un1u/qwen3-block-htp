#!/usr/bin/env python3
"""Prepare a real layer-14 GQA-group package for EXP-0042.

The generated integer reference deliberately mirrors the two-stage HMX
carrier contract: power-of-two HMX conversion to a centered U8 temporary,
followed by a small integer multiplier.  QK emits a log2 score grid and AV
emits the existing asymmetric U8 attention-concat qparam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


ABI_VERSION = 1
M = 64
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
HMX_CENTER = 128
EXP_FRAC_BITS = 15
MAX_MULTIPLIER = 18
MAX_SHIFT = 15
DIVISION_MODES = {"exact": 1, "sole": 2, "endpoint": 3}
CONFIG = struct.Struct("<4I5i6I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
        ),
    )
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--fraction-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--division", choices=tuple(DIVISION_MODES), default="sole"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/"
            "attention_group4_f3_sole"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0042"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_carrier(ratio: float) -> tuple[int, int, float]:
    best: tuple[float, int, int, float] | None = None
    for shift in range(MAX_SHIFT + 1):
        for multiplier in range(1, MAX_MULTIPLIER + 1):
            effective = math.ldexp(float(multiplier), -shift)
            error = abs(effective - ratio)
            candidate = (error, shift, multiplier, effective)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[2], best[1], best[3]


def centered_hmx_requant(
    accumulator: np.ndarray,
    multiplier: int,
    shift: int,
    output_zero_point: int,
) -> tuple[np.ndarray, int]:
    divisor = 1 << shift
    rounding = divisor // 2 if shift else 0
    intermediate = np.floor_divide(
        accumulator.astype(np.int64) + HMX_CENTER * divisor + rounding,
        divisor,
    )
    intermediate_clipped = np.clip(intermediate, 0, 255)
    saturations = int(np.count_nonzero(intermediate != intermediate_clipped))
    output = (
        (intermediate_clipped.astype(np.int64) - HMX_CENTER) * multiplier
        + output_zero_point
    )
    output_clipped = np.clip(output, 0, 255)
    saturations += int(np.count_nonzero(output != output_clipped))
    return output_clipped.astype(np.uint8), saturations


def divide_probability(
    exponent: int, total: int, mode: str, valid_count: int
) -> int:
    weight = 1 << (EXP_FRAC_BITS - exponent)
    if valid_count == 1:
        return 255
    if mode == "exact":
        return min(255, (weight * 255 + total // 2) // total)
    leading = total.bit_length() - 1
    next_bit = (total >> (leading - 1)) & 1 if leading else 0
    if mode == "sole":
        coefficient = 145 if next_bit else 209
    else:
        coefficient = 171 if next_bit else 256
    shift = exponent + leading - EXP_FRAC_BITS
    numerator = 255 * coefficient
    denominator = 256
    if shift >= 0:
        denominator <<= shift
    else:
        numerator <<= -shift
    return min(255, (numerator + denominator // 2) // denominator)


def log2_softmax(
    scores: np.ndarray, fraction_bits: int, division: str
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(scores.shape, dtype=np.uint8)
    exponent_codes = np.full(scores.shape, 15, dtype=np.uint8)
    rounding = 1 << (fraction_bits - 1)
    for head in range(scores.shape[0]):
        for row in range(M):
            valid = row + 1
            maximum = int(scores[head, row, :valid].max())
            delta = maximum - scores[head, row, :valid].astype(np.int32)
            exponents = np.minimum(
                15, (delta + rounding) >> fraction_bits
            ).astype(np.uint8)
            exponent_codes[head, row, :valid] = exponents
            total = sum(1 << (EXP_FRAC_BITS - int(value)) for value in exponents)
            for column, exponent in enumerate(exponents):
                probability[head, row, column] = divide_probability(
                    int(exponent), total, division, valid
                )
    return probability, exponent_codes


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    left = actual.astype(np.float64).reshape(-1)
    right = reference.astype(np.float64).reshape(-1)
    difference = left - right
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": float(np.dot(left, right) / denominator)
        if denominator > 0.0
        else 0.0,
    }


def write_bytes(staging: Path, name: str, data: bytes) -> dict[str, object]:
    path = staging / name
    path.write_bytes(data)
    return {"file": name, "bytes": len(data), "sha256": sha256(path)}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not 0 <= args.group < KV_HEADS:
        raise ValueError(f"group must be in [0,{KV_HEADS - 1}]")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="package-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing"
    try:
        source_manifest_path = source / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        qparams = source_manifest["u8_qparams"]
        q_all = np.fromfile(
            source / "reference_w4u8_q_rope_u8.bin", dtype=np.uint8
        ).reshape(M, HEADS, HEAD_DIM)
        k_all = np.fromfile(
            source / "reference_w4u8_k_rope_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM)
        v_all = np.fromfile(
            source / "reference_w4u8_v_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM)

        first_q_head = args.group * Q_HEADS_PER_GROUP
        q = np.ascontiguousarray(
            q_all[:, first_q_head : first_q_head + Q_HEADS_PER_GROUP, :]
            .transpose(1, 0, 2)
        )
        k = np.ascontiguousarray(k_all[:, args.group, :])
        v = np.ascontiguousarray(v_all[:, args.group, :])

        q_qp = qparams["q_rope"]
        k_qp = qparams["k_rope"]
        v_qp = qparams["v"]
        p_qp = qparams["attention_probability"]
        y_qp = qparams["attention_concat"]
        k_centered = k.astype(np.int16) - int(k_qp["zero_point"])
        if k_centered.min() < -128 or k_centered.max() > 127:
            raise ValueError("K cannot be represented exactly as S8")
        k_s8 = k_centered.astype(np.int8)
        v_centered = v.astype(np.int32) - int(v_qp["zero_point"])
        # V is an HMX RHS operand.  A per-GQA-group calibrated symmetric
        # carrier preserves the observed asymmetric-U8 range instead of
        # wasting S8 codes on the unreachable theoretical [0,255] endpoints.
        v_denominator = max(
            1, int(np.max(np.abs(v_centered.astype(np.int64))))
        )
        v_numerator = 127
        v_signed = np.where(
            v_centered >= 0,
            (v_centered * v_numerator + v_denominator // 2)
            // v_denominator,
            -((-v_centered * v_numerator + v_denominator // 2)
              // v_denominator),
        )
        if v_signed.min() < -128 or v_signed.max() > 127:
            raise ValueError("V recenter overflow")
        v_s8 = v_signed.astype(np.int8)
        v_s8_scale = (
            float(v_qp["scale"]) * v_denominator / v_numerator
        )

        score_step = math.log(2.0) / (1 << args.fraction_bits)
        score_ratio = (
            float(q_qp["scale"])
            * float(k_qp["scale"])
            / math.sqrt(HEAD_DIM)
            / score_step
        )
        score_multiplier, score_shift, score_effective = choose_carrier(
            score_ratio
        )
        av_ratio = (
            float(p_qp["scale"]) * v_s8_scale / float(y_qp["scale"])
        )
        av_multiplier, av_shift, av_effective = choose_carrier(av_ratio)

        q_centered = q.astype(np.int32) - int(q_qp["zero_point"])
        score_accumulator = np.matmul(
            q_centered, k_s8.astype(np.int32).T
        )
        score, score_saturations = centered_hmx_requant(
            score_accumulator,
            score_multiplier,
            score_shift,
            HMX_CENTER,
        )
        probability, exponent_codes = log2_softmax(
            score, args.fraction_bits, args.division
        )
        av_accumulator = np.matmul(
            probability.astype(np.int32), v_s8.astype(np.int32)
        )
        integer_output, av_saturations = centered_hmx_requant(
            av_accumulator,
            av_multiplier,
            av_shift,
            int(y_qp["zero_point"]),
        )

        q_real = q_centered.astype(np.float64) * float(q_qp["scale"])
        k_real = k_centered.astype(np.float64) * float(k_qp["scale"])
        v_real = v_centered.astype(np.float64) * float(v_qp["scale"])
        real_scores = np.matmul(q_real, k_real.T) / math.sqrt(HEAD_DIM)
        real_scores = np.where(
            np.triu(np.ones((M, M), dtype=bool), 1)[None, :, :],
            -np.inf,
            real_scores,
        )
        real_max = np.max(real_scores, axis=-1, keepdims=True)
        real_exp = np.exp(real_scores - real_max)
        real_probability = real_exp / np.sum(real_exp, axis=-1, keepdims=True)
        real_output = np.matmul(real_probability, v_real)
        standard_output_u8 = np.clip(
            np.rint(real_output / float(y_qp["scale"]))
            + int(y_qp["zero_point"]),
            0,
            255,
        ).astype(np.uint8)
        standard_probability_u8 = np.clip(
            np.rint(real_probability * 255.0), 0, 255
        ).astype(np.uint8)

        files: dict[str, dict[str, object]] = {}
        config_bytes = CONFIG.pack(
            ABI_VERSION,
            args.group,
            args.fraction_bits,
            DIVISION_MODES[args.division],
            int(q_qp["zero_point"]),
            int(k_qp["zero_point"]),
            int(v_qp["zero_point"]),
            int(p_qp["zero_point"]),
            int(y_qp["zero_point"]),
            v_numerator,
            v_denominator,
            score_shift,
            score_multiplier,
            av_shift,
            av_multiplier,
        )
        files["attention_config"] = write_bytes(
            staging, "attention_config.bin", config_bytes
        )
        for name, array in (
            ("q_group_u8", q),
            ("k_group_u8", k),
            ("v_group_u8", v),
            ("reference_score_u8", score),
            ("reference_probability_u8", probability),
            ("reference_output_u8", integer_output),
            ("diagnostic_exponent_code_u8", exponent_codes),
            ("diagnostic_standard_probability_u8", standard_probability_u8),
            ("diagnostic_standard_output_u8", standard_output_u8),
        ):
            files[name] = write_bytes(
                staging, f"{name}.bin", np.ascontiguousarray(array).tobytes()
            )

        manifest = {
            "experiment": "EXP-0042",
            "package_abi": ABI_VERSION,
            "source": str(source),
            "source_manifest_sha256": sha256(source_manifest_path),
            "layer": 14,
            "sequence_length": M,
            "group": args.group,
            "q_heads": [first_q_head, first_q_head + 1],
            "kv_head": args.group,
            "head_dim": HEAD_DIM,
            "fraction_bits": args.fraction_bits,
            "division": args.division,
            "qparams": {
                "q": q_qp,
                "k": k_qp,
                "v": v_qp,
                "probability": p_qp,
                "output": y_qp,
                "v_s8_scale": v_s8_scale,
                "score_log2_step": score_step,
            },
            "integer_carrier": {
                "score": {
                    "target_ratio": score_ratio,
                    "multiplier": score_multiplier,
                    "shift": score_shift,
                    "effective_ratio": score_effective,
                    "relative_error": abs(score_effective - score_ratio)
                    / score_ratio,
                },
                "av": {
                    "target_ratio": av_ratio,
                    "multiplier": av_multiplier,
                    "shift": av_shift,
                    "effective_ratio": av_effective,
                    "relative_error": abs(av_effective - av_ratio) / av_ratio,
                },
                "v_recenter": {
                    "numerator": v_numerator,
                    "denominator": v_denominator,
                    "minimum": int(v_s8.min()),
                    "maximum": int(v_s8.max()),
                },
            },
            "diagnostics": {
                "score_saturations": score_saturations,
                "av_saturations": av_saturations,
                "probability_vs_standard_u8": metrics(
                    probability, standard_probability_u8
                ),
                "output_vs_standard_u8": metrics(
                    integer_output, standard_output_u8
                ),
                "probability_row_sum_min": int(
                    probability.sum(axis=-1).min()
                ),
                "probability_row_sum_max": int(
                    probability.sum(axis=-1).max()
                ),
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if publish.exists():
            shutil.rmtree(publish)
        shutil.copytree(staging, publish)
        publish.rename(output)
        shutil.rmtree(staging)
        print(json.dumps(manifest["diagnostics"], sort_keys=True))
        print(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if publish.exists():
            shutil.rmtree(publish, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
