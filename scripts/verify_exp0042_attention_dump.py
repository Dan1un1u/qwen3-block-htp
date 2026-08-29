#!/usr/bin/env python3
"""Verify EXP-0042 integer Attention from its actual device inputs.

The six dump tensors are copied out only by an audit-on execution.  Q/K/V are
the declared input boundary of the integer Attention core.  This verifier
independently recomputes QK requantization, log2 Softmax, and AV requantization
from those bytes and requires every device stage to match exactly.  Comparisons
against retained W4U8 and W4F16 tensors are reported separately as diagnostic
quantization accuracy and never replace the implementation reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from prepare_exp0042_attention import (  # noqa: E402
    CONFIG,
    HMX_CENTER,
    centered_hmx_requant,
    log2_softmax,
)

M = 64
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
DIVISION_NAMES = {1: "exact", 2: "sole", 3: "endpoint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(path: Path, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype=np.uint8)
    if value.size != count:
        raise ValueError(f"{path}: got {value.size} bytes, expected {count}")
    return value


def unpack_feature_tiles(path: Path, heads: int) -> np.ndarray:
    physical = read_exact(path, heads * M * HEAD_DIM).reshape(
        heads, HEAD_DIM // 32, M, 32
    )
    return np.ascontiguousarray(
        physical.transpose(2, 0, 1, 3).reshape(M, heads, HEAD_DIM)
    )


def unpack_score_tiles(path: Path) -> np.ndarray:
    physical = read_exact(path, HEADS * M * M).reshape(
        HEADS, M // 32, M, 32
    )
    return np.ascontiguousarray(
        physical.transpose(0, 2, 1, 3).reshape(HEADS, M, M)
    )


def unpack_av_tiles(path: Path) -> np.ndarray:
    physical = read_exact(path, HEADS * M * HEAD_DIM).reshape(
        HEADS, HEAD_DIM // 32, M, 32
    )
    return np.ascontiguousarray(
        physical.transpose(0, 2, 1, 3).reshape(HEADS, M, HEAD_DIM)
    )


def exact_difference(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = actual.astype(np.int16) - reference.astype(np.int16)
    mismatch = np.flatnonzero(difference)
    result: dict[str, object] = {
        "elements": int(actual.size),
        "mismatches": int(mismatch.size),
        "max_lsb": int(np.max(np.abs(difference))),
        "mean_absolute_lsb": float(np.mean(np.abs(difference))),
    }
    if mismatch.size:
        flat = int(mismatch[0])
        result["first_mismatch_index"] = [
            int(value) for value in np.unravel_index(flat, actual.shape)
        ]
        result["first_actual"] = int(actual.flat[flat])
        result["first_reference"] = int(reference.flat[flat])
    return result


def real_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    left = actual.astype(np.float64).reshape(-1)
    right = reference.astype(np.float64).reshape(-1)
    difference = left - right
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    reference_rms = float(np.sqrt(np.mean(right * right)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / reference_rms if reference_rms else 0.0,
        "cosine": float(np.dot(left, right) / denominator)
        if denominator
        else 0.0,
    }


def dequantize(value: np.ndarray, qparam: dict[str, object]) -> np.ndarray:
    return (
        value.astype(np.float64) - float(qparam["zero_point"])
    ) * float(qparam["scale"])


def main() -> int:
    args = parse_args()
    dump = args.dump.resolve()
    package = args.package.resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    reference = (
        args.reference.resolve()
        if args.reference is not None
        else Path(manifest["reference_source"]).resolve()
    )
    reference_manifest_path = reference / "manifest.json"
    reference_manifest_sha256 = sha256(reference_manifest_path)
    if reference_manifest_sha256 != manifest["reference_manifest_sha256"]:
        raise ValueError("reference manifest hash differs from package provenance")
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    qparams = reference_manifest["u8_qparams"]

    q = unpack_feature_tiles(dump / "actual_q_tiles_u8.bin", HEADS)
    k = unpack_feature_tiles(dump / "actual_k_tiles_u8.bin", KV_HEADS)
    v = unpack_feature_tiles(dump / "actual_v_tiles_u8.bin", KV_HEADS)
    actual_score = unpack_score_tiles(dump / "actual_score_tiles_u8.bin")
    actual_probability = unpack_score_tiles(
        dump / "actual_probability_tiles_u8.bin"
    )
    actual_av = unpack_av_tiles(dump / "actual_av_tiles_u8.bin")

    expected_score = np.empty_like(actual_score)
    expected_probability = np.empty_like(actual_probability)
    expected_av = np.empty_like(actual_av)
    score_saturations = 0
    av_saturations = 0
    v_recenter_saturations = 0
    config_bytes = (package / "attention_config_all_groups.bin").read_bytes()
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError("unexpected Attention config byte count")

    for group in range(KV_HEADS):
        fields = CONFIG.unpack_from(config_bytes, group * CONFIG.size)
        abi, config_group, fraction_bits, division_mode, *values = fields
        (
            q_zero_point,
            k_zero_point,
            v_zero_point,
            _probability_zero_point,
            output_zero_point,
            v_numerator,
            v_denominator,
            score_shift,
            score_multiplier,
            av_shift,
            av_multiplier,
        ) = values
        if abi != 1 or config_group != group or division_mode not in DIVISION_NAMES:
            raise ValueError(f"invalid Attention config record {group}")

        first_head = group * Q_HEADS_PER_GROUP
        q_group = np.ascontiguousarray(
            q[:, first_head : first_head + Q_HEADS_PER_GROUP, :].transpose(
                1, 0, 2
            )
        )
        k_head = k[:, group, :]
        v_head = v[:, group, :]
        score_accumulator = np.matmul(
            q_group.astype(np.int32) - q_zero_point,
            (k_head.astype(np.int32) - k_zero_point).T,
        )
        score, count = centered_hmx_requant(
            score_accumulator,
            score_multiplier,
            score_shift,
            HMX_CENTER,
        )
        score_saturations += count
        probability, _ = log2_softmax(
            score, fraction_bits, DIVISION_NAMES[division_mode]
        )

        centered_v = v_head.astype(np.int32) - v_zero_point
        recentered_v = np.where(
            centered_v >= 0,
            (centered_v * v_numerator + v_denominator // 2)
            // v_denominator,
            -(
                (-centered_v * v_numerator + v_denominator // 2)
                // v_denominator
            ),
        )
        v_recenter_saturations += int(
            np.count_nonzero((recentered_v < -128) | (recentered_v > 127))
        )
        signed_v = np.clip(recentered_v, -128, 127).astype(np.int8)
        av_accumulator = np.matmul(
            probability.astype(np.int32), signed_v.astype(np.int32)
        )
        av, count = centered_hmx_requant(
            av_accumulator,
            av_multiplier,
            av_shift,
            output_zero_point,
        )
        av_saturations += count
        expected_score[first_head : first_head + Q_HEADS_PER_GROUP] = score
        expected_probability[
            first_head : first_head + Q_HEADS_PER_GROUP
        ] = probability
        expected_av[first_head : first_head + Q_HEADS_PER_GROUP] = av

    exact = {
        "qk": exact_difference(actual_score, expected_score),
        "log2_softmax": exact_difference(
            actual_probability, expected_probability
        ),
        "av": exact_difference(actual_av, expected_av),
    }
    mask_violations = 0
    for row in range(M):
        mask_violations += int(
            np.count_nonzero(actual_probability[:, row, row + 1 :])
        )
    core_exact = all(stage["mismatches"] == 0 for stage in exact.values())

    retained_u8 = {
        "q_rope": np.fromfile(
            reference / "reference_w4u8_q_rope_u8.bin", dtype=np.uint8
        ).reshape(M, HEADS, HEAD_DIM),
        "k_rope": np.fromfile(
            reference / "reference_w4u8_k_rope_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "v": np.fromfile(
            reference / "reference_w4u8_v_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "attention_concat": np.fromfile(
            reference / "reference_w4u8_attention_concat_u8.bin",
            dtype=np.uint8,
        ).reshape(M, HEADS, HEAD_DIM),
    }
    actual_logical = {
        "q_rope": q,
        "k_rope": k,
        "v": v,
        "attention_concat": actual_av.transpose(1, 0, 2),
    }
    retained_diagnostic = {
        name: exact_difference(actual_logical[name], retained_u8[name])
        for name in actual_logical
    }

    w4f16 = {
        "q_rope": np.fromfile(
            reference / "reference_w4f16_q_rope_f16.bin", dtype="<f2"
        ).reshape(M, HEADS, HEAD_DIM),
        "k_rope": np.fromfile(
            reference / "reference_w4f16_k_rope_f16.bin", dtype="<f2"
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "v": np.fromfile(
            reference / "reference_w4f16_v_f16.bin", dtype="<f2"
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "attention_probability": np.fromfile(
            reference / "reference_w4f16_attention_probability_f16.bin",
            dtype="<f2",
        ).reshape(HEADS, M, M),
        "attention_concat": np.fromfile(
            reference / "reference_w4f16_attention_concat_f16.bin",
            dtype="<f2",
        ).reshape(M, HEADS, HEAD_DIM),
    }
    real_actual = {
        name: dequantize(actual_logical[name], qparams[name])
        for name in actual_logical
    }
    real_actual["attention_probability"] = dequantize(
        actual_probability, qparams["attention_probability"]
    )
    w4f16_diagnostic = {
        name: real_metrics(real_actual[name], w4f16[name])
        for name in w4f16
    }

    result = {
        "experiment": "EXP-0042",
        "implementation_reference_boundary": "actual device U8 Q/K/V tensors",
        "implementation_reference": "independent host integer QK, log2 Softmax, and AV",
        "core_exact": core_exact,
        "physical_audit_note": (
            "dump DDR writes exist only in audit-on and are excluded from "
            "the audit-off zero-intermediate-DDR performance contract"
        ),
        "dump_sha256": {
            path.name: sha256(path) for path in sorted(dump.glob("*.bin"))
        },
        "exact_stage_comparison": exact,
        "probability_mask_violations": mask_violations,
        "score_saturations": score_saturations,
        "v_recenter_saturations": v_recenter_saturations,
        "av_saturations": av_saturations,
        "retained_w4u8_code_diagnostic": retained_diagnostic,
        "w4f16_accuracy_diagnostic_only": w4f16_diagnostic,
        "reference_manifest_sha256": reference_manifest_sha256,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if core_exact and mask_violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
