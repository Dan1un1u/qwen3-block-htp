#!/usr/bin/env python3
"""Prove whether EXP-0076 can preserve the existing AV byte mapping.

The accepted implementation first asks HMX to round the accumulator to an
unsigned carrier centred at 128 and then applies the integer AV multiplier in
HVX.  EXP-0076 proposes one affine HMX conversion instead.  This script
evaluates both mappings on the retained real layer-14 Attention tensors and
checks the accepted two-stage implementation against its archived reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


CONFIG = struct.Struct("<15I")
HEADS = 16
KV_HEADS = 8
Q_HEADS_PER_GROUP = 2
ROWS = 64
HEAD_DIM = 128
HMX_CENTER = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Override the reference_source recorded by the package manifest.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_u8(value: int) -> int:
    return 0 if value < 0 else 255 if value > 255 else value


def round_div_signed(value: int, denominator: int) -> int:
    if value >= 0:
        return (value + denominator // 2) // denominator
    return -((-value + denominator // 2) // denominator)


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = (
        args.reference.resolve()
        if args.reference is not None
        else Path(manifest["reference_source"]).resolve()
    )

    probability_path = (
        package / "reference_w4u8_integer_attention_probability_u8.bin"
    )
    accepted_av_path = (
        package / "reference_w4u8_integer_attention_heads_u8.bin"
    )
    v_path = reference / "reference_w4u8_v_u8.bin"
    config_path = package / "attention_config_all_groups.bin"

    probability = probability_path.read_bytes()
    accepted_av = accepted_av_path.read_bytes()
    v_codes = v_path.read_bytes()
    configs = config_path.read_bytes()
    if len(probability) != HEADS * ROWS * ROWS:
        raise ValueError("unexpected probability tensor size")
    if len(accepted_av) != HEADS * ROWS * HEAD_DIM:
        raise ValueError("unexpected accepted AV tensor size")
    if len(v_codes) != ROWS * KV_HEADS * HEAD_DIM:
        raise ValueError("unexpected V tensor size")
    if len(configs) != KV_HEADS * CONFIG.size:
        raise ValueError("unexpected Attention config size")

    groups: list[dict[str, object]] = []
    accepted_reference_mismatches = 0
    accepted_reference_max_lsb = 0
    fused_mismatches = 0
    fused_max_lsb = 0
    fused_abs_error_sum = 0
    element_count = 0

    for group in range(KV_HEADS):
        fields = CONFIG.unpack_from(configs, group * CONFIG.size)
        (
            abi,
            config_group,
            _fraction_bits,
            _division_mode,
            _q_zero_point,
            _k_zero_point,
            v_zero_point,
            _probability_zero_point,
            output_zero_point,
            v_numerator,
            v_denominator,
            _score_shift,
            _score_multiplier,
            av_shift,
            av_multiplier,
        ) = fields
        if abi != 1 or config_group != group:
            raise ValueError(f"invalid config record {group}")

        signed_v = [[0] * HEAD_DIM for _ in range(ROWS)]
        for token in range(ROWS):
            for channel in range(HEAD_DIM):
                source_index = (
                    (token * KV_HEADS + group) * HEAD_DIM + channel
                )
                centered = v_codes[source_index] - v_zero_point
                requantized = round_div_signed(
                    centered * v_numerator, v_denominator
                )
                signed_v[token][channel] = max(-128, min(127, requantized))

        divisor = 1 << av_shift
        group_accepted_mismatches = 0
        group_accepted_max_lsb = 0
        group_fused_mismatches = 0
        group_fused_max_lsb = 0
        group_fused_abs_error_sum = 0
        accumulator_min: int | None = None
        accumulator_max: int | None = None

        for local_head in range(Q_HEADS_PER_GROUP):
            head = group * Q_HEADS_PER_GROUP + local_head
            for row in range(ROWS):
                probability_base = (head * ROWS + row) * ROWS
                output_base = (head * ROWS + row) * HEAD_DIM
                for channel in range(HEAD_DIM):
                    accumulator = sum(
                        probability[probability_base + token]
                        * signed_v[token][channel]
                        for token in range(ROWS)
                    )
                    accumulator_min = (
                        accumulator
                        if accumulator_min is None
                        else min(accumulator_min, accumulator)
                    )
                    accumulator_max = (
                        accumulator
                        if accumulator_max is None
                        else max(accumulator_max, accumulator)
                    )

                    # Accepted two-stage mapping: the first rounding and U8
                    # saturation happen before the multiplier.
                    intermediate = clip_u8(
                        (accumulator + HMX_CENTER * divisor + divisor // 2)
                        // divisor
                    )
                    accepted = clip_u8(
                        (intermediate - HMX_CENTER) * av_multiplier
                        + output_zero_point
                    )

                    # Proposed one-stage affine mapping: multiplier and zero
                    # point are folded into the HMX conversion before its sole
                    # rounding/saturation point.
                    fused = clip_u8(
                        (accumulator * av_multiplier
                         + output_zero_point * divisor
                         + divisor // 2)
                        // divisor
                    )
                    archived = accepted_av[output_base + channel]
                    accepted_error = abs(accepted - archived)
                    fused_error = abs(fused - accepted)
                    if accepted_error:
                        group_accepted_mismatches += 1
                    if fused_error:
                        group_fused_mismatches += 1
                    group_accepted_max_lsb = max(
                        group_accepted_max_lsb, accepted_error
                    )
                    group_fused_max_lsb = max(
                        group_fused_max_lsb, fused_error
                    )
                    group_fused_abs_error_sum += fused_error

        group_elements = Q_HEADS_PER_GROUP * ROWS * HEAD_DIM
        accepted_reference_mismatches += group_accepted_mismatches
        accepted_reference_max_lsb = max(
            accepted_reference_max_lsb, group_accepted_max_lsb
        )
        fused_mismatches += group_fused_mismatches
        fused_max_lsb = max(fused_max_lsb, group_fused_max_lsb)
        fused_abs_error_sum += group_fused_abs_error_sum
        element_count += group_elements
        groups.append(
            {
                "group": group,
                "av_multiplier": av_multiplier,
                "av_shift": av_shift,
                "output_zero_point": output_zero_point,
                "accumulator_min": accumulator_min,
                "accumulator_max": accumulator_max,
                "accepted_vs_archived_mismatches":
                    group_accepted_mismatches,
                "accepted_vs_archived_max_lsb": group_accepted_max_lsb,
                "one_stage_vs_accepted_mismatches": group_fused_mismatches,
                "one_stage_vs_accepted_max_lsb": group_fused_max_lsb,
                "one_stage_vs_accepted_mean_abs_lsb":
                    group_fused_abs_error_sum / group_elements,
                "global_exact_affine_representability": av_multiplier == 1,
            }
        )

    gate_pass = (
        accepted_reference_mismatches == 0
        and fused_mismatches == 0
        and fused_max_lsb == 0
    )
    result = {
        "experiment": "EXP-0076",
        "failure_boundary": "pre_build_integer_correctness_gate",
        "accepted_two_stage_reference": {
            "mismatches": accepted_reference_mismatches,
            "max_lsb": accepted_reference_max_lsb,
        },
        "one_stage_hmx_affine_candidate": {
            "mismatches": fused_mismatches,
            "elements": element_count,
            "mismatch_percent": 100.0 * fused_mismatches / element_count,
            "max_lsb": fused_max_lsb,
            "mean_abs_lsb": fused_abs_error_sum / element_count,
        },
        "proof_note": (
            "For AV multiplier M>1, the accepted mapping is "
            "M*round(accumulator/2^shift)+zero_point. Its plateaus are "
            "2^shift wide and adjacent outputs differ by M. One affine HMX "
            "conversion has only one rounding point and instead implements "
            "round(M*accumulator/2^shift+zero_point); it cannot reproduce "
            "the accepted staircase globally."
        ),
        "groups": groups,
        "correctness_gate": "pass" if gate_pass else "fail",
        "profiling": {
            "status": "unavailable",
            "reason": (
                "The mandatory byte-exact numerical gate failed before a "
                "candidate runtime was built or executed."
            ),
        },
        "provenance": {
            "package": str(package),
            "reference": str(reference),
            "manifest_sha256": sha256(manifest_path),
            "config_sha256": sha256(config_path),
            "probability_sha256": sha256(probability_path),
            "v_codes_sha256": sha256(v_path),
            "accepted_av_sha256": sha256(accepted_av_path),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
