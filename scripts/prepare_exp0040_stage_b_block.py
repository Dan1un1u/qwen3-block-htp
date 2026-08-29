#!/usr/bin/env python3
"""Create the isolated EXP-0040 complete-block package.

The block host constructs the interleaved Gate/Up and Down streaming bundles
from the original W4 carriers plus the historical W4U8 HMX bias words.  The
package therefore only adds the exact 65,536-entry SiLU×Up LUT to the retained
EXP-0022 block inputs, references and projection tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
        ),
    )
    parser.add_argument(
        "--mlp-source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/"
            "mlp_package_layer14_m64_output_requant_v5_round"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0040/"
            "block_package_layer14_m64_stage_b"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0040"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    common = [
        "block_input_f16.bin",
        "reference_w4f16_block_output_f16.bin",
        "reference_w4u8_block_input_u8.bin",
        "reference_w4u8_block_output_u8.bin",
        "qparams_u8.bin",
        "input_norm_weight_f16.bin",
        "post_norm_weight_f16.bin",
        "q_norm_weight_f16.bin",
        "k_norm_weight_f16.bin",
        "rope_cos_f16.bin",
        "rope_sin_f16.bin",
    ]
    projections = ["q", "k", "v", "o", "gate", "up", "down"]
    files = common + [
        filename
        for projection in projections
        for filename in (
            f"{projection}_weight_w4_hmx.bin",
            f"{projection}_weight_w4_scale_f32.bin",
        )
    ]
    lut_source = args.mlp_source / "silu_up_lut_u16.bin"
    for path in [args.block_source / "manifest.json", lut_source] + [
        args.block_source / filename for filename in files
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing package: {args.output}")

    args.staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stage-b-block-", dir=args.staging_root))
    try:
        records: dict[str, dict[str, object]] = {}
        for filename in files:
            source = args.block_source / filename
            destination = staging / filename
            shutil.copy2(source, destination)
            records[filename] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source": "EXP-0022 retained block package",
            }
        shutil.copy2(lut_source, staging / lut_source.name)
        records[lut_source.name] = {
            "bytes": (staging / lut_source.name).stat().st_size,
            "sha256": sha256(staging / lut_source.name),
            "source": "EXP-0040 Stage-A exact U8-pair LUT",
        }
        manifest = {
            "experiment": "EXP-0040",
            "stage": "B",
            "execution_unit": "qwen3_layer14_complete_block_m64",
            "block_source": str(args.block_source),
            "block_source_manifest_sha256": sha256(
                args.block_source / "manifest.json"
            ),
            "mlp_source": str(args.mlp_source),
            "mlp_source_manifest_sha256": sha256(
                args.mlp_source / "manifest.json"
            ),
            "candidate_contract": {
                "changed_scope": "W4U8 Gate/Up, exact LUT handoff, and Down only",
                "historical_hmx_conversion": True,
                "historical_complete_output_required_byte_exact": True,
                "intermediate_ddr_allowed": False,
                "vtcm_request_bytes": 8 * 1024 * 1024,
                "runtime_bundle_construction": (
                    "host interleaves retained Gate/Up W4 tiles and copies the "
                    "retained HMX bias words; no new quantization parameters"
                ),
            },
            "files": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, args.output)
        print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
